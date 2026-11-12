"""Paired Qwen-only CI repair development run. No retrieval or code execution."""

import argparse
import importlib.metadata
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import compare_raw_edit_models as comparison

raw = comparison.raw
pilot = raw.pilot
from paper_mistral_baseline import relevant_logs


ARMS = ("legacy_plain", "raw_plain")
MODEL = pilot.BENCHMARK / ".venv/qwen-coder-model"
PRIOR = pilot.BENCHMARK / "out/raw-edit-model-comparison/6eefc4788c3efc1fd144"
COHORT = pilot.BENCHMARK / "out/edit-format-pilot/0c6e365def079719ceb5"
CONTEXT = 8192
CAP = 2048
TEMPERATURE = 0.2


def prompts_for(row, blocks):
    logs = relevant_logs(row["logs"])
    raw_prompt = raw.prompt_for(logs, blocks)
    prefix = f"Repair the code. Check output: {logs}\n\n"
    return {
        "legacy_plain": pilot.make_prompt({"logs": row["logs"]}, blocks),
        "raw_plain": prefix + raw_prompt.removeprefix(prefix).replace(
            "followed by raw Python on subsequent lines", "followed by raw replacement text on subsequent lines", 1),
    }


def replacements_for(answer, blocks, arm):
    error = answer.get("skip")
    if not error:
        try:
            if answer["finish_reason"] != "stop":
                raise ValueError("output_token_limit")
            parse = raw.parse_answer if arm == "raw_plain" else pilot.parse_response
            return parse(answer["response"], blocks), None
        except ValueError as exc:
            error = str(exc)
    return [block["old"] for block in blocks], error


def summarize(rows, n):
    summary = {}
    for arm in ARMS:
        values = [row for row in rows if row["arm"] == arm]
        no_regression = [m for m in values if m["nonempty_patch"] and m["syntax_preserved"]]
        summary[arm] = {
            "scored": len(values), "expected": n,
            "valid_response": sum(m["error"] is None for m in values),
            "nonempty_patch": sum(m["nonempty_patch"] for m in values),
            "nonempty_no_detected_syntax_regression": len(no_regression),
            "multi_block_nonempty_no_detected_syntax_regression": sum(m["target_blocks"] > 1 for m in no_regression),
            "edit_token_f1": statistics.mean(m["edit_token_f1"] for m in values) if values else None,
            "exact_reference": sum(m["patch_exact_match"] for m in values),
            "errors": dict(Counter(m["error"] for m in values if m["error"])),
        }
    complete = all(arm["scored"] == n for arm in summary.values())
    legacy, candidate = [summary[a] for a in ARMS]
    gate = None if not complete else (
        candidate["nonempty_no_detected_syntax_regression"] > legacy["nonempty_no_detected_syntax_regression"]
        and candidate["edit_token_f1"] >= legacy["edit_token_f1"]
        and candidate["multi_block_nonempty_no_detected_syntax_regression"] > 0
    )
    return {"scope": "Offline text/syntax proxies, not CI Pass@1", "complete": complete,
            "development_gate_passed": gate, "arms": summary, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=pilot.BENCHMARK / "out/qwen-ci-dev")
    parser.add_argument("--audit-only", action="store_true", help="Re-score saved answers without the GPU")
    args = parser.parse_args()
    from datasets import Dataset

    synthetic = pilot.read_json(PRIOR / "run_metadata.json")
    cohort = pilot.read_json(COHORT / "run_metadata.json")
    if PRIOR.name != pilot.digest(synthetic)[:20] or COHORT.name != pilot.digest(cohort)[:20]:
        raise ValueError("Prior run manifest mismatch")
    if synthetic["protocol_source"] != pilot.file_sha(Path(raw.__file__)):
        raise ValueError("Raw edit protocol changed")
    if not pilot.read_json(PRIOR / "metrics.json")["complete"]:
        raise ValueError("Incomplete synthetic comparison")
    data, = [p for p in (pilot.BENCHMARK / ".venv/dataset-cache").rglob("*test.arrow")
             if pilot.file_sha(p) == cohort["dataset_sha256"]]
    rows = {row["id"]: row for row in Dataset.from_file(str(data)) if row["id"] in cohort["ids"]}
    prepared = {}
    for row in rows.values():
        gold = pilot.diff_blocks(row["diff"])
        blocks = pilot.prediction_blocks(gold)
        files = pilot.target_files(row, blocks, pilot.BENCHMARK / ".venv/rag-repos")
        prepared[row["id"]] = (gold, blocks, files, prompts_for(row, blocks))
    modules = [__file__, raw.__file__, raw.previous.__file__, pilot.__file__]
    # Pin the imported log extraction and offline scoring code as well.
    modules.extend([sys.modules["paper_mistral_baseline"].__file__, sys.modules["evaluation"].__file__,
                    sys.modules["retrieval"].__file__])
    manifest = {"model": synthetic["model"], "revision": synthetic["revision"],
                "model_hashes": synthetic["qwen_hashes"], "packages": synthetic["packages"],
                "source_hashes": {Path(m).name: pilot.file_sha(Path(m)) for m in modules},
                "synthetic_run": PRIOR.name, "cohort": COHORT.name, "dataset_sha256": cohort["dataset_sha256"],
                "failing_shas": {str(i): rows[i]["sha_fail"] for i in cohort["ids"]},
                "ids": cohort["ids"], "seeds": cohort["seeds"], "arms": ARMS,
                "context": CONTEXT, "output_cap": CAP, "temperature": TEMPERATURE, "retrieval": False}
    folder = args.out / pilot.digest(manifest)[:20]
    print(f"Qwen no-retrieval development run: {folder}", flush=True)

    if not args.audit_only:
        for name, sha in manifest["model_hashes"].items():
            if pilot.file_sha(MODEL / name) != sha:
                raise ValueError(f"Changed local checkpoint file: {name}")
        sys.path.insert(0, str(raw.previous.DEPS))
        if {k: importlib.metadata.version(k) for k in manifest["packages"]} != manifest["packages"]:
            raise ValueError("Model runtime differs from synthetic run")
        from llguidance import LLMatcher, LLTokenizer
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = load(str(MODEL))
        eos = tokenizer.eos_token_id
        if eos not in tokenizer.eos_token_ids:
            raise ValueError("Qwen EOS mismatch")
        n_vocab = pilot.read_json(MODEL / "config.json")["vocab_size"]
        backend = LLTokenizer(tokenizer.backend_tokenizer.to_str(), n_vocab=n_vocab, eos_token=eos)
        pilot.save_json(folder / "run_metadata.json", manifest)
    elif pilot.digest(pilot.read_json(folder / "run_metadata.json")) != pilot.digest(manifest):
        raise ValueError("Saved development manifest mismatch")

    metrics = []
    n = len(cohort["ids"]) * len(cohort["seeds"])
    for i in cohort["ids"]:
        gold, blocks, files, prompts = prepared[i]
        if not args.audit_only:
            token_sets = {arm: tokenizer.apply_chat_template([{"role": "user", "content": prompts[arm]}],
                                                             add_generation_prompt=True, tokenize=True)
                          for arm in ARMS}
            grammar = raw.grammar_for(len(blocks))
        for seed in cohort["seeds"]:
            for arm in ARMS:
                path = folder / "examples" / str(i) / str(seed) / f"{arm}.json"
                identity = {"run": folder.name, "id": i, "seed": seed, "arm": arm, "sha_fail": rows[i]["sha_fail"]}
                if path.exists() or args.audit_only:
                    answer, sha = pilot.read_checked(path)
                    if answer["identity"] != identity or answer["prompt"] != prompts[arm]:
                        raise ValueError("Saved answer identity mismatch")
                else:
                    tokens = token_sets[arm]
                    response, finish, count = "", None, 0
                    started = time.monotonic()
                    skip = "input_budget" if len(tokens) + CAP + 256 > CONTEXT else None
                    if not skip:
                        processors = ([raw.previous.TokenConstraint(LLMatcher(backend, grammar), len(tokens), eos)]
                                      if arm == "raw_plain" else [])
                        mx.random.seed(seed + i)
                        for piece in stream_generate(model, tokenizer, tokens, max_tokens=CAP,
                                                      sampler=make_sampler(temp=TEMPERATURE),
                                                      logits_processors=processors):
                            response += piece.text
                            finish, count = piece.finish_reason, piece.generation_tokens
                        mx.clear_cache()
                    answer = {"identity": identity, "prompt": prompts[arm], "prompt_tokens": len(tokens),
                              "response": response, "finish_reason": finish, "generation_tokens": count,
                              "skip": skip, "elapsed_seconds": time.monotonic() - started}
                    sha = pilot.digest(answer)
                    pilot.save_json(path, {**answer, "payload_sha256": sha})
                edits, error = replacements_for(answer, blocks, arm)
                values = pilot.measure(files, gold, {"replacements": edits})
                metrics.append({**identity, "error": error, "prediction_sha256": sha,
                                "target_blocks": len(blocks), "changed_blocks": sum(b["old"] != e for b, e in zip(blocks, edits)),
                                "prompt_tokens": answer["prompt_tokens"],
                                "generation_tokens": answer["generation_tokens"], **values})
                if not args.audit_only:
                    pilot.save_json(folder / "metrics.json", summarize(metrics, n))
                    print(f"{i}/{seed}/{arm}: {error or 'valid'} "
                          f"nonempty={values['nonempty_patch']} F1={values['edit_token_f1']:.3f}", flush=True)
    result = summarize(metrics, n)
    if args.audit_only and result != pilot.read_json(folder / "metrics.json"):
        raise ValueError("Recomputed development metrics differ")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
