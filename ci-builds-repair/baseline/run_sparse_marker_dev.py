"""Compare optional plain-text block replacements against the saved non-RAG control."""

import argparse
import importlib.metadata
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import run_edit_format_pilot as pilot
from retrieval import source_lines


ARM_NAMES = ("legacy_plain", "sparse_markers_plain")
INSTRUCTION = (
    "\n\nReturn only changed blocks with their exact start and end markers. "
    "Omit unchanged blocks. Put only replacement code inside markers. "
    "If nothing changes, return [no edits].\n\n"
)
MARKER = re.compile(r"\[(start|end) of (.+)\]")
FENCE = re.compile(r"```[A-Za-z0-9_+.-]*")


def marker_prompt(base):
    if base.count(pilot.LEGACY) != 1:
        raise ValueError("Unexpected legacy prompt layout")
    return base.replace(pilot.LEGACY, INSTRUCTION, 1)


def parse_markers(response, blocks):
    """Accept a nonempty subset of exact, balanced markers or an explicit no-op."""
    lines = source_lines(response.strip())
    if lines and FENCE.fullmatch(lines[0].rstrip("\r\n")) and lines[-1] == "```":
        lines = lines[1:-1]
    if "\0" in response:
        raise ValueError("invalid_replacement")
    if "".join(lines).strip() == "[no edits]":
        return [b["old"] for b in blocks]
    tags = {f"{b['path']}#L{b['start']}#B{i}": i for i, b in enumerate(blocks)}
    replacements = [b["old"] for b in blocks]
    opened, body, seen = None, [], set()
    for line in lines:
        mark = MARKER.fullmatch(line.rstrip("\r\n"))
        if mark:
            kind, tag = mark.groups()
            if tag not in tags:
                raise ValueError("unknown_block_marker")
            if kind == "start":
                if opened is not None or tag in seen:
                    raise ValueError("nested_or_duplicate_block")
                opened, body = tag, []
            else:
                if opened != tag:
                    raise ValueError("unmatched_end_marker")
                i = tags[tag]
                replacement = "".join(body)
                if replacement and blocks[i]["old"] and not blocks[i]["old"].endswith(("\r", "\n")):
                    replacement = re.sub(r"(?:\r\n|\n|\r)\Z", "", replacement, count=1)
                replacements[i] = replacement
                seen.add(tag)
                opened = None
        elif opened is not None:
            body.append(line)
        elif line.strip():
            raise ValueError("unexpected_text_outside_blocks")
    if opened is not None:
        raise ValueError("unclosed_block")
    if not seen:
        raise ValueError("missing_edit_markers")
    return replacements


def summarize(rows, expected):
    arms = {}
    for arm in ARM_NAMES:
        cases = [row for row in rows if row["arm"] == arm]
        good = [m for m in cases if m["nonempty_patch"] and m["syntax_preserved"]]
        arms[arm] = {
            "evaluated": len(cases),
            "valid_response": sum(m["status"] == "applied" for m in cases),
            "nonempty_patch": sum(m["nonempty_patch"] for m in cases),
            "nonempty_no_detected_syntax_regression": len(good),
            "multi_block_nonempty_no_detected_syntax_regression": sum(m["target_blocks"] > 1 for m in good),
            "partial_edit": sum(m["partial_edit"] for m in cases),
            "edit_token_f1": statistics.mean(m["edit_token_f1"] for m in cases) if cases else None,
            "exact_reference": sum(m["patch_exact_match"] for m in cases),
            "errors": dict(Counter(m["error"] for m in cases if m["error"])),
        }
    complete = all(arm["evaluated"] == expected for arm in arms.values())
    legacy, sparse = [arms[a] for a in ARM_NAMES]
    gate = None if not complete else (
        sparse["nonempty_no_detected_syntax_regression"] > legacy["nonempty_no_detected_syntax_regression"]
        and sparse["edit_token_f1"] >= legacy["edit_token_f1"]
        and sparse["multi_block_nonempty_no_detected_syntax_regression"] > 0
    )
    return {"complete": complete, "expected_per_arm": expected, "arms": arms,
            "development_gate_passed": gate, "scope": "Offline proxies, not CI Pass@1", "cases": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    benchmark = pilot.BENCHMARK
    p.add_argument("--controls", type=Path, default=benchmark / "out/rag-local/2d9349bb16247814246b")
    p.add_argument("--prior-pilot", type=Path, default=benchmark / "out/edit-format-pilot/0c6e365def079719ceb5")
    p.add_argument("--out", type=Path, default=benchmark / "out/sparse-marker-dev")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    from datasets import Dataset

    old = pilot.read_json(args.controls / "run_metadata.json")
    dev = pilot.read_json(args.prior_pilot / "run_metadata.json")
    if (args.controls.name != pilot.digest(old)[:20] or args.prior_pilot.name != pilot.digest(dev)[:20]
        or dev["control_run"] != args.controls.name or dev["dataset_sha256"] != old["dataset_sha256"]):
        raise ValueError("Unexpected control or development cohort")
    for name, sha in old["source_hashes"].items():
        location = benchmark if name == "paper_mistral_baseline.py" else pilot.ROOT.parent / "rag"
        if pilot.file_sha(location / name) != sha:
            raise ValueError(f"Changed control source: {name}")
    data, = [p for p in (benchmark / ".venv/dataset-cache").rglob("*test.arrow")
             if pilot.file_sha(p) == dev["dataset_sha256"]]
    rows = {r["id"]: r for r in Dataset.from_file(str(data)) if r["id"] in dev["ids"]}
    model_path = benchmark / ".venv/mistral-model"
    for name, sha in dev["model_hashes"].items():
        if pilot.file_sha(model_path / name) != sha:
            raise ValueError(f"Changed model file: {name}")
    if {name: importlib.metadata.version(name) for name in dev["packages"]} != dev["packages"]:
        raise ValueError("Changed model runtime")
    manifest = {"control": args.controls.name, "development_cohort": args.prior_pilot.name,
                "ids": dev["ids"], "seeds": dev["seeds"], "arms": ARM_NAMES,
                "model_hashes": dev["model_hashes"], "packages": dev["packages"],
                "settings": {k: old[k] for k in ("temperature", "context_window", "max_tokens")},
                "source_sha256": pilot.file_sha(Path(__file__)), "helper_sha256": pilot.file_sha(Path(pilot.__file__))}
    folder = args.out / pilot.digest(manifest)[:20]
    pilot.save_json(folder / "run_metadata.json", manifest)
    print(f"Run: {folder}; examples={dev['ids']}; seeds={dev['seeds']}; retrieval disabled", flush=True)
    if args.prepare_only:
        return
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(str(model_path))
    cap = old["max_tokens"]
    budget = old["context_window"] - cap - 256
    metrics = []
    for i in dev["ids"]:
        row = rows[i]
        gold = pilot.diff_blocks(row["diff"])
        blocks = pilot.prediction_blocks(gold)
        files = pilot.target_files(row, blocks, benchmark / ".venv/rag-repos")
        controls = pilot.controls_for(row, gold, files, args.controls, dev["seeds"])
        for seed in dev["seeds"]:
            prior, sha, _ = controls[seed, "baseline"]
            if sha != dev["control_checksums"][str(i)][f"{seed}/baseline"]:
                raise ValueError("Changed control answer")
            for arm in ARM_NAMES:
                identity = {"id": i, "seed": seed, "arm": arm, "control_sha256": sha, "run": folder.name}
                path = folder / "examples" / str(i) / str(seed) / f"{arm}.json"
                if path.exists():
                    answer, _ = pilot.read_checked(path)
                    if answer["identity"] != identity:
                        raise ValueError("Cached identity mismatch")
                elif arm == "legacy_plain":
                    answer = {**prior, "identity": identity, "reused_control": True}
                else:
                    prompt = marker_prompt(prior["prompt"])
                    def tokenize(text):
                        return tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                             add_generation_prompt=True, tokenize=True)

                    if len(tokenize(prior["prompt"])) != prior["prompt_tokens"]:
                        raise ValueError("Control prompt token count changed")
                    tokens = tokenize(prompt)
                    started = time.monotonic()
                    response, finish, count = "", None, 0
                    error = None
                    if len(tokens) > budget:
                        error = "oracle_prompt_exceeds_budget"
                    else:
                        mx.random.seed(seed + i)
                        for piece in stream_generate(model, tokenizer, tokens, max_tokens=cap,
                                                     sampler=make_sampler(temp=old["temperature"])):
                            response += piece.text
                            finish, count = piece.finish_reason, piece.generation_tokens
                        if finish == "length":
                            error = "output_token_limit"
                    replacements = [b["old"] for b in blocks]
                    if error is None:
                        try:
                            replacements = parse_markers(response, blocks)
                        except ValueError as problem:
                            error = str(problem)
                    answer = {"identity": identity, "reused_control": False, "prompt": prompt,
                              "prompt_tokens": len(tokens), "response": response, "finish_reason": finish,
                              "generation_tokens": count, "replacements": replacements, "error": error,
                              "status": "unapplied" if error else "applied",
                              "packed_chunks": [], "elapsed_seconds": time.monotonic() - started}
                    mx.clear_cache()
                if not path.exists():
                    pilot.save_json(path, {**answer, "payload_sha256": pilot.digest(answer)})
                changed = sum(b["old"] != replacement for b, replacement in zip(blocks, answer["replacements"]))
                metric = {"id": i, "seed": seed, "arm": arm, "changed_blocks": changed,
                          "target_blocks": len(blocks), "partial_edit": 0 < changed < len(blocks),
                          **{k: answer[k] for k in ("status", "error", "prompt_tokens", "generation_tokens")},
                          **pilot.measure(files, gold, answer)}
                metrics.append(metric)
                pilot.save_json(folder / "metrics.json", summarize(metrics, len(dev["ids"]) * len(dev["seeds"])))
                print(f"{i}/{seed}/{arm}: {answer['error'] or 'valid'} "
                      f"nonempty={metric['nonempty_patch']} F1={metric['edit_token_f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
