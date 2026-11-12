"""Paired legacy-vs-sparse edit-format pilot; no network or project execution."""

import argparse
import hashlib
import importlib.metadata
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from verify_completed_run import BENCHMARK, ROOT, apply_edits, diff_blocks, digest
from verify_completed_run import make_prompt, parse_response, prediction_blocks, save_json, score


LEGACY = (
    "\n\nEdit the following code blocks. Return every block exactly once with its "
    "start and end markers; put only replacement code between them. Do not add commentary.\n\n"
)
SPARSE = (
    '\n\nReturn only JSON {"edits":{"B0":"replacement code"}}. Use block IDs. '
    'Omit unchanged blocks. Preserve indentation.\n\n'
)
ARMS = ("legacy_plain", "legacy_rag", "sparse_plain", "sparse_rag")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sparse_prompt(base):
    # Replace only the output instruction, not a matching string in the log or code.
    match = re.search(re.escape(LEGACY) + r"(?=\[start of )", base)
    if not match or base.count(LEGACY) != 1:
        raise ValueError("Unexpected legacy prompt layout")
    return base[:match.start()] + SPARSE + base[match.end():]


def unique_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def parse_sparse(response, blocks):
    response = response.strip()
    fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", response, re.DOTALL)
    if fence:
        response = fence[1]
    try:
        result = json.loads(response, object_pairs_hook=unique_keys)
    except json.JSONDecodeError as error:
        raise ValueError("invalid_json") from error
    if not isinstance(result, dict) or set(result) != {"edits"} or not isinstance(result["edits"], dict):
        raise ValueError("invalid_schema")
    edits = result["edits"]
    allowed = {f"B{i}": b for i, b in enumerate(blocks)}
    if not edits.keys() <= allowed.keys():
        raise ValueError("unknown_block_id")
    for key, value in edits.items():
        if not isinstance(value, str) or "\0" in value:
            raise ValueError("invalid_replacement")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("invalid_replacement") from error
        old = allowed[key]["old"]
        if value and old.endswith(("\r", "\n")) and not value.endswith(("\r", "\n")):
            edits[key] = value + ("\r\n" if old.endswith("\r\n") else old[-1])
    return [edits.get(key, block["old"]) for key, block in allowed.items()]


def read_checked(path):
    record = read_json(path)
    checksum = record.pop("payload_sha256")
    if digest(record) != checksum:
        raise ValueError(f"Bad prediction checksum: {path}")
    return record, checksum


def measure(files, gold, prediction):
    predicted = apply_edits(files, prediction_blocks(gold), prediction["replacements"])
    metrics = score(files, gold, predicted, [], [])
    return {key: value for key, value in metrics.items() if not key.startswith("retrieval_")}


def controls_for(row, gold, files, run, seeds):
    blocks = prediction_blocks(gold)
    base = make_prompt({"logs": row["logs"]}, blocks)
    source = read_json(run / "retrieval" / f"{row['id']}.json")["provenance"]
    if source["sha_fail"] != row["sha_fail"]:
        raise ValueError("Historical snapshot mismatch")
    controls = {}
    for seed in seeds:
        for method in ("baseline", "rag"):
            path = run / "examples" / str(row["id"]) / f"seed_{seed}" / method
            pred, sha = read_checked(path / "prediction.json")
            metric = read_json(path / "metrics.json")
            identity = {"id": row["id"], "seed": seed, "run": run.name, "method": method,
                        "snapshot": source["files_sha256"]}
            if pred["identity"] != identity or metric["prediction_sha256"] != sha:
                raise ValueError("Historical prediction identity mismatch")
            if not pred["prompt"].endswith(base) or (
                method == "baseline" and (pred["prompt"] != base or pred["packed_chunks"])
            ):
                raise ValueError("Historical prompt mismatch")
            replacements = (parse_response(pred["response"], blocks) if pred["status"] == "applied"
                            else [b["old"] for b in blocks])
            if replacements != pred["replacements"] or any(
                metric[k] != v for k, v in measure(files, gold, pred).items()
            ):
                raise ValueError("Historical scores or replacements mismatch")
            prefix = pred["prompt"][:-len(base)]
            controls[seed, method] = (pred, sha, prefix + sparse_prompt(base))
    rag_evidence = [digest([controls[s, "rag"][0][k] for k in ("prompt", "packed_chunks")]) for s in seeds]
    if len(set(rag_evidence)) != 1:
        raise ValueError("Historical packed evidence varies by seed")
    return controls


def target_files(row, blocks, cache):
    from git import Repo

    name = f"{row['repo_owner']}__{row['repo_name']}.git"
    if not re.fullmatch(r"[\w.-]+", name) or not re.fullmatch(r"[0-9a-f]{40}", row["sha_fail"]):
        raise ValueError("Invalid cached source identity")
    files = {}
    with Repo(cache / name) as repo:
        tree = repo.commit(row["sha_fail"]).tree
        for b in blocks:
            if b["path"] not in files:
                try:
                    blob = tree / b["path"]
                except KeyError:
                    if b["count"]:
                        raise
                else:
                    files[b["path"]] = blob.data_stream.read().decode("utf-8")
    apply_edits(files, blocks, [b["old"] for b in blocks])
    return files


def summarize(metrics, expected):
    arms = {}
    for arm in ARMS:
        rows = [m for m in metrics if m["arm"] == arm]
        arms[arm] = {
            "evaluated": len(rows), "expected": expected,
            "valid_response": sum(m["status"] == "applied" for m in rows),
            "nonempty_patch": sum(m["nonempty_patch"] for m in rows),
            "partial_edit": sum(m["partial_edit"] for m in rows),
            "exact_reference": sum(m["patch_exact_match"] for m in rows),
            "edit_token_f1": statistics.mean(m["edit_token_f1"] for m in rows) if rows else None,
            "syntax_regression_cases": sum(bool(m["syntax_regression_files"]) for m in rows),
            "errors": dict(Counter(m["error"] for m in rows if m["error"])),
        }
    complete = all(a["evaluated"] == expected for a in arms.values())
    contrasts = {}
    for key in ("edit_token_f1", "nonempty_patch", "valid_response"):
        paired = {(m["id"], m["seed"], m["arm"]): (
            int(m["status"] == "applied") if key == "valid_response" else m[key]
        ) for m in metrics}
        values = []
        for identifier, seed in sorted({(i, s) for i, s, _ in paired}):
            if all((identifier, seed, a) in paired for a in ARMS):
                lp, lr, sp, sr = [paired[identifier, seed, a] for a in ARMS]
                values.append((sp-lp, sr-lr, lr-lp, sr-sp, (sr-sp)-(lr-lp)))
        labels = ("format_plain", "format_rag", "retrieval_legacy", "retrieval_sparse", "interaction")
        contrasts[key] = {"paired_cases": len(values), **{
            label: statistics.mean(v[i] for v in values) if values else None
            for i, label in enumerate(labels)
        }}
    return {"scope": "Offline proxies; not CI Pass@1", "complete": complete,
            "arms": arms, "matched_contrasts": contrasts, "cases": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, default=BENCHMARK / "out/rag-local/2d9349bb16247814246b")
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--model-path", type=Path, default=BENCHMARK / ".venv/mistral-model")
    parser.add_argument("--repo-cache", type=Path, default=BENCHMARK / ".venv/rag-repos")
    parser.add_argument("--out", type=Path, default=BENCHMARK / "out/edit-format-pilot")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    from datasets import Dataset

    old = read_json(args.controls / "run_metadata.json")
    if args.controls.name != digest(old)[:20]:
        raise ValueError("Control run manifest mismatch")
    for name, sha in old["source_hashes"].items():
        location = BENCHMARK if name == "paper_mistral_baseline.py" else ROOT.parent / "rag"
        if file_sha(location / name) != sha:
            raise ValueError(f"Control source changed: {name}")
    data = args.dataset_file
    if data is None:
        candidates = list((BENCHMARK / ".venv/dataset-cache").rglob("*test.arrow"))
        matches = [p for p in candidates if file_sha(p) == old["dataset_sha256"]]
        if len(matches) != 1:
            parser.error("Supply --dataset-file for the original Arrow dataset")
        data = matches[0]
    if file_sha(data) != old["dataset_sha256"]:
        raise ValueError("Dataset does not match controls")
    # Fixed outcome-blind selection; don't use earlier scores to pick pilot cases.
    ids = sorted(old["ids"], key=lambda i: digest(["compact-edit-pilot-v1", i]))[:6]
    seeds = old["seeds"]
    rows = {r["id"]: r for r in Dataset.from_file(str(data)) if r["id"] in ids}
    cases = {}
    history = {}
    for identifier in ids:
        row = rows[identifier]
        gold = diff_blocks(row["diff"])
        files = target_files(row, prediction_blocks(gold), args.repo_cache)
        saved = controls_for(row, gold, files, args.controls, seeds)
        cases[identifier] = (gold, files, saved)
        history[str(identifier)] = {f"{s}/{m}": sha for (s, m), (_, sha, _) in saved.items()}
    runtime = read_json(ROOT.parent / "rag/results/runtime_identity.json")
    model_hashes = {"config.json": old["model_config_sha256"], **runtime["tokenizer_sha256"],
                    "model.safetensors": runtime["model_weights_sha256"]}
    for file, sha in model_hashes.items():
        if file_sha(args.model_path / file) != sha:
            raise ValueError(f"Local model differs from control: {file}")
    packages = {name: importlib.metadata.version(name) for name in runtime["packages"]}
    if packages != runtime["packages"]:
        raise ValueError("Runtime versions differ from the control")
    manifest = {"version": 1, "control_run": args.controls.name, "control_checksums": history,
                "ids": ids, "seeds": seeds, "arms": ARMS, "model": old["model_id"],
                "dataset_sha256": old["dataset_sha256"], "model_hashes": model_hashes,
                "packages": packages, "source_sha256": file_sha(Path(__file__)),
                **{k: old[k] for k in ("temperature", "context_window", "max_tokens")}}
    folder = args.out / digest(manifest)[:20]
    save_json(folder / "run_metadata.json", manifest)
    print(f"Pilot: {folder}; IDs={ids}; seeds={seeds}; {2*len(ids)*len(seeds)} fresh generations planned", flush=True)
    if args.prepare_only:
        return
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(str(args.model_path))
    budget = old["context_window"] - old["max_tokens"] - 256

    def tokenize(text):
        return tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                             add_generation_prompt=True, tokenize=True)

    metrics = []
    for identifier in ids:
        gold, files, saved = cases[identifier]
        blocks = prediction_blocks(gold)
        for seed in seeds:
            for method, suffix in (("baseline", "plain"), ("rag", "rag")):
                prior, sha, prompt = saved[seed, method]
                if len(tokenize(prior["prompt"])) != prior["prompt_tokens"]:
                    raise ValueError("Control prompt token count changed")
                for variant in ("legacy", "sparse"):
                    arm = f"{variant}_{suffix}"
                    path = folder / "examples" / str(identifier) / str(seed) / f"{arm}.json"
                    identity = {"id": identifier, "seed": seed, "arm": arm, "control_sha256": sha,
                                "run": folder.name}
                    if path.exists():
                        pred, _ = read_checked(path)
                        if pred["identity"] != identity:
                            raise ValueError("Pilot cache identity mismatch")
                    elif variant == "legacy":
                        pred = {**prior, "identity": identity, "reused_control": True}
                    else:
                        tokens = tokenize(prompt)
                        started = time.monotonic()
                        response, finish, count = "", None, 0
                        error = None
                        if len(tokens) > budget:
                            error = "oracle_prompt_exceeds_budget"
                        else:
                            mx.random.seed(seed + identifier)
                            for piece in stream_generate(model, tokenizer, tokens, max_tokens=old["max_tokens"],
                                                         sampler=make_sampler(temp=old["temperature"])):
                                response += piece.text
                                finish, count = piece.finish_reason, piece.generation_tokens
                            if finish == "length":
                                error = "output_token_limit"
                        replacements = [b["old"] for b in blocks]
                        if error is None:
                            try:
                                replacements = parse_sparse(response, blocks)
                            except ValueError as problem:
                                error = str(problem)
                        pred = {"identity": identity, "reused_control": False, "prompt": prompt,
                                "prompt_tokens": len(tokens), "response": response, "finish_reason": finish,
                                "generation_tokens": count, "replacements": replacements, "error": error,
                                "status": "unapplied" if error else "applied",
                                "packed_chunks": prior["packed_chunks"], "elapsed_seconds": time.monotonic()-started}
                        mx.clear_cache()
                    if not path.exists():
                        save_json(path, {**pred, "payload_sha256": digest(pred)})
                    changed = sum(b["old"] != value for b, value in zip(blocks, pred["replacements"]))
                    result = {"id": identifier, "seed": seed, "arm": arm,
                              "changed_blocks": changed, "target_blocks": len(blocks),
                              "partial_edit": 0 < changed < len(blocks),
                              **{k: pred[k] for k in ("status", "error", "prompt_tokens", "generation_tokens")},
                              **measure(files, gold, pred)}
                    metrics.append(result)
                    save_json(folder / "metrics.json", summarize(metrics, len(ids)*len(seeds)))
                    print(f"{identifier}/{seed}/{arm}: {pred['error'] or 'valid'} "
                          f"nonempty={result['nonempty_patch']} F1={result['edit_token_f1']:.3f}", flush=True)
    print(f"Completed {len(metrics)} scored pilot cases", flush=True)


if __name__ == "__main__":
    main()
