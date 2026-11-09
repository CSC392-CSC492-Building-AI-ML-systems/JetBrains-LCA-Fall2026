"""Run a single issue, training pilot or all 50 py/test issues."""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

from bug_localization.agentic.loop import Limits, predict, prepare_input
from bug_localization.baseline_eval.run_all_tfidf import DATASET
from bug_localization.long_context.run import checked_write, implementation_identity
from bug_localization.rag.file_ranking import complete_ranking, validate_response
from bug_localization.rag.model import ModelSettings
from bug_localization.rag.prepare_repositories import CANDIDATE_POLICY, validate_settings
from bug_localization.rag.ranking_prompt import TokenCounter
from bug_localization.rag.retrieval import digest, project_example
from bug_localization.rag.run_ranking import read_checked
from bug_localization.rag.run_retrieval import load_dataset_input, prediction_columns, save_json


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "runs"


def validate_saved(result, manifest, signature):
    index = result["example_index"]
    if (result["run_signature"] != signature or index not in manifest["example_indices"]
            or digest(result["example_input"]) != manifest["input_sha256"][str(index)]
            or result["limits"] != manifest["tool_limits"]
            or result["repo_token_budget"] != manifest["repo_token_budget"]):
        raise ValueError("Prediction identity mismatch")
    baseline = result["bm25_ranking"]
    allowed = [r["file"] for r in baseline]
    if (sorted(allowed) != result["packing"]["allowed_candidate_paths"]
            or len(allowed) != len(set(allowed)) or len(allowed) != result["candidate_count"]):
        raise ValueError("Candidate mismatch")
    status = result["status"]
    success = status in {"success", "repaired"}
    fallback = status in {"fallback_context", "fallback_request", "fallback_validation"}
    if status != "dry_run" and not (success or fallback):
        raise ValueError("Incomplete result")
    if result["bm25_fallback"] != fallback or result["model_ranking_success"] != success:
        raise ValueError("Outcome mismatch")
    expected = baseline
    if success:
        validation = validate_response(result["ranking_response"], allowed)
        if validation != result["validation"] or validation["status"] != status:
            raise ValueError("Validation mismatch")
        expected = complete_ranking(validation["ranking"], baseline)
    if result["ranking"] != expected:
        raise ValueError("Ranking mismatch")
    if result["tool_calls"] > manifest["tool_limits"]["tool_calls"]:
        raise ValueError("Tool budget exceeded")
    for turn in result["trace"]:
        if (digest(turn["messages"]) != turn["messages_sha256"]
                or turn["input_tokens_estimated"] > manifest["model"]["context_window"]
                - manifest["model"]["max_output_tokens"]
                - max(manifest["model"]["safety_margin"], math.ceil(manifest["model"]["context_window"] * .05))):
            raise ValueError("Invalid trace request")


def run_examples(dataset, repo_token_budget=4096, output_root=DEFAULT_OUTPUT, settings=ModelSettings(),
                 start=0, limit=1, split="train", window=40, stride=20, limits=Limits(),
                 dry_run=False, resume=True, client=None, counter=None, dataset_name=DATASET):
    settings.validate()
    limits.validate()
    validate_settings(window, stride)
    if (type(repo_token_budget) is not int or repo_token_budget < 0 or start < 0
            or start >= len(dataset) or limit <= 0):
        raise ValueError("Require nonnegative source budget, valid start and positive limit")
    indices = list(range(start, min(start + limit, len(dataset))))
    inputs = prediction_columns(dataset)
    examples = {i: project_example(inputs[i]) for i in indices}
    counter = counter or TokenCounter(settings.model_name, settings.tokenizer)
    implementation = implementation_identity()
    implementation["agentic_source_hashes"] = {p.name: digest(p.read_text(encoding="utf-8"))
                                                for p in sorted(ROOT.glob("*.py"))}
    manifest = {"experiment_version": 1, "configuration": "py", "mode": "agentic",
                "dataset": dataset_name, "split": split, "dataset_total_examples": len(dataset),
                "dataset_fingerprint": getattr(dataset, "_fingerprint", None), "example_indices": indices,
                "input_sha256": {str(i): digest(e) for i, e in examples.items()},
                "repo_token_budget": repo_token_budget, "model": settings.identity(),
                "token_counter": counter.identity, "candidate_policy": CANDIDATE_POLICY,
                "tool_limits": asdict(limits), "dry_run": dry_run,
                "retrieval": {"window": window, "stride": stride, "k1": 1.2, "b": .75,
                              "evidence_limit": "all positive matching chunks before token packing"},
                **implementation}
    signature = digest(manifest)
    folder = Path(output_root).resolve() / signature
    save_json(folder / "run_metadata.json", manifest)
    save_json(folder.parent / "latest_run.json", {"run_directory": signature})
    summary = {"run_signature": signature, "configuration": "py", "split": split, "mode": "agentic",
               "dry_run": dry_run, "total_examples": len(indices), "dataset_total_examples": len(dataset),
               "example_indices": indices, "processed": 0, "successful": 0, "failed": 0, "reused": 0,
               "remaining": len(indices), "result_files": [], "failures": [], "status_counts": {}}
    save_json(folder / "summary.json", summary)
    for index in indices:
        path = folder / "examples" / f"{index:03d}.json"
        started = time.perf_counter()
        base = {"run_signature": signature, "example_index": index, "example_input": examples[index],
                "mode": "agentic", "repo_token_budget": repo_token_budget}
        try:
            result = None
            if resume:
                try:
                    cached = read_checked(path)
                    validate_saved(cached, manifest, signature)
                    if cached["status"] in {"success", "repaired"} | ({"dry_run"} if dry_run else set()):
                        result = cached
                        summary["reused"] += 1
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if result is None:
                checked_write(path, {**base, "status": "running"})
                prepared = prepare_input(examples[index], window, stride)
                base.update(resolved_base_sha=prepared["resolved_base_sha"],
                            snapshot_sha256=prepared["snapshot_sha256"], candidate_count=len(prepared["files"]))
                result = {**base, **predict(prepared, repo_token_budget, settings, counter, limits,
                                           client, dry_run, window, stride),
                          "elapsed_seconds": time.perf_counter() - started}
                validate_saved(result, manifest, signature)
                checked_write(path, result)
            summary["successful"] += 1
            summary["result_files"].append(path.relative_to(folder).as_posix())
            status = result["status"]
            summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
            print(f"[{index}] {status}: {result['tool_calls']} tools; stop={result['stop_reason']}", flush=True)
        except Exception as error:
            failure = {**base, "status": "failed", "error_type": type(error).__name__,
                       "elapsed_seconds": time.perf_counter() - started}
            checked_write(path, failure)
            summary["failed"] += 1
            summary["failures"].append({"example_index": index, "error_type": type(error).__name__})
            print(f"[{index}] failed: {type(error).__name__}", file=sys.stderr, flush=True)
        summary["processed"] += 1
        summary["remaining"] -= 1
        save_json(folder / "summary.json", summary)
    print(f"Outputs: {folder}", flush=True)
    return folder / "summary.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", dest="model_name")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--repo-token-budget", type=int, default=4096)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--stride", type=int, default=20)
    for name, field in Limits.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=field.default)
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_bytes()) if args.config else {}
        if args.model_name:
            config["model_name"] = args.model_name
        settings = ModelSettings(**config)
        settings.validate()
        limits = Limits(**{k: getattr(args, k) for k in Limits.__dataclass_fields__})
        limits.validate()
        dataset = load_dataset_input(args.split, args.dataset_cache_dir, args.dataset_file)
        path = run_examples(dataset, args.repo_token_budget, args.output_dir, settings, args.start,
                            args.limit, args.split, args.window, args.stride, limits, args.dry_run, args.resume)
        summary = json.loads(path.read_bytes())
        return int(bool(summary["failed"] or any(k.startswith("fallback_") for k in summary["status_counts"])))
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
