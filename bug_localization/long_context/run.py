"""Run direct, rag or issue_only; defaults to all 50 Python test examples."""

import argparse
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from bug_localization.baseline_eval.run_all_tfidf import DATASET
from bug_localization.long_context.experiment import MODEL_STATUSES, predict, prepare_input, validate_saved
from bug_localization.long_context.prompt import MODES
from bug_localization.rag.model import ModelSettings
from bug_localization.rag.prepare_repositories import CANDIDATE_POLICY, validate_settings
from bug_localization.rag.ranking_prompt import TokenCounter
from bug_localization.rag.retrieval import digest, project_example
from bug_localization.rag.run_ranking import read_checked
from bug_localization.rag.run_retrieval import load_dataset_input, prediction_columns, save_json


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "runs"


def implementation_identity():
    root = ROOT.parents[1]
    paths = list(ROOT.glob("*.py"))
    paths += [root / "bug_localization" / "rag" / name for name in (
        "prepare_repositories.py", "bm25_index.py", "retrieval.py", "retrieval_tokenizer.py",
        "ranking_prompt.py", "model.py", "file_ranking.py", "run_retrieval.py", "run_ranking.py")]
    paths += [root / "bug_localization" / "baseline_eval" / name
              for name in ("run_all_tfidf.py", "evaluate_tfidf.py")]
    paths.append(root / "project_level_code_completion" / "rag" / "bm25.py")
    packages = {}
    for name in ("openai", "tiktoken", "datasets", "GitPython"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"source_hashes": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(paths)}, "packages": packages, "python": platform.python_version()}


def checked_write(path, value):
    save_json(path, {**value, "payload_sha256": digest(value)})


def run_examples(dataset, mode, repo_token_budget, output_root=DEFAULT_OUTPUT, settings=ModelSettings(),
                 start=0, limit=50, split="test", window=40, stride=20, dry_run=False,
                 resume=True, client=None, counter=None, dataset_name=DATASET):
    settings.validate()
    validate_settings(window, stride)
    if (mode not in MODES or type(repo_token_budget) is not int or repo_token_budget < 0
            or (mode == "issue_only" and repo_token_budget != 0)):
        raise ValueError("Known mode and nonnegative source budget required; issue_only requires 0")
    if start < 0 or limit <= 0 or start >= len(dataset):
        raise ValueError("Require a valid start index and positive limit")
    selected = list(range(start, min(len(dataset), start + limit)))
    counter = counter or TokenCounter(settings.model_name, settings.tokenizer)
    inputs = prediction_columns(dataset)
    examples = {i: project_example(inputs[i]) for i in selected}
    manifest = {
        "experiment_version": 1, "configuration": "py", "split": split, "dataset": dataset_name,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None), "dataset_total_examples": len(dataset),
        "example_indices": selected, "input_sha256": {str(i): digest(row) for i, row in examples.items()},
        "mode": mode, "repo_token_budget": repo_token_budget, "model": settings.identity(),
        "token_counter": counter.identity, "candidate_policy": CANDIDATE_POLICY,
        "retrieval": {"window": window, "stride": stride, "k1": 1.2, "b": .75,
                      "evidence_limit": "all positive matching chunks before token packing"},
        "dry_run": dry_run, **implementation_identity(),
    }
    signature = digest(manifest)
    folder = Path(output_root).resolve() / signature
    save_json(folder / "run_metadata.json", manifest)
    save_json(folder.parent / "latest_run.json", {"run_directory": signature})
    summary = {
        "run_signature": signature, "configuration": "py", "split": split, "mode": mode,
        "repo_token_budget": repo_token_budget, "total_examples": len(selected),
        "dataset_total_examples": len(dataset), "example_indices": selected, "processed": 0,
        "remaining": len(selected), "successful": 0, "failed": 0, "reused": 0,
        "result_files": [], "failures": [], "status_counts": {}, "dry_run": dry_run,
    }
    save_json(folder / "summary.json", summary)
    for index in selected:
        path = folder / "examples" / f"{index:03d}.json"
        started = time.perf_counter()
        stage = "snapshot_and_retrieval"
        base = {"run_signature": signature, "example_index": index, "example_input": examples[index],
                "mode": mode, "repo_token_budget": repo_token_budget}
        try:
            result = None
            if resume:
                try:
                    cached = read_checked(path)
                    validate_saved(cached, manifest, signature)
                    if cached["status"] in MODEL_STATUSES | ({"dry_run"} if dry_run else set()):
                        result = cached
                        summary["reused"] += 1
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if result is None:
                prepared = prepare_input(examples[index], window, stride)
                base.update(resolved_base_sha=prepared["resolved_base_sha"],
                            snapshot_sha256=prepared["snapshot_sha256"],
                            candidate_count=len(prepared["files"]))
                stage = "packing_and_prediction"
                # Persist before any paid request; interrupted requests are never
                # mistaken for completed model rankings on resume.
                checked_write(path, {**base, "status": "running"})
                result = {**base, **predict(prepared, mode, repo_token_budget, settings, counter, client, dry_run),
                          "elapsed_seconds": time.perf_counter() - started}
                validate_saved(result, manifest, signature)
                checked_write(path, result)
            summary["result_files"].append(path.relative_to(folder).as_posix())
            summary["successful"] += 1  # Completed artifacts, including explicit fallbacks/dry runs.
            status = result["status"]
            summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
            print(f"[{index}] {status}: source={result['packing']['actual_source_tokens']}/{repo_token_budget} "
                  f"prompt={result['packing']['total_prompt_tokens']}", flush=True)
        except Exception as error:
            failure = {**base, "status": "failed", "stage": stage, "error_type": type(error).__name__,
                       "requested_source_tokens": repo_token_budget, "actual_source_tokens": None,
                       "total_prompt_tokens": None, "request_latency_seconds": None,
                       "elapsed_seconds": time.perf_counter() - started}
            checked_write(path, failure)
            summary["failed"] += 1
            summary["failures"].append({"example_index": index, "stage": stage,
                                        "error_type": type(error).__name__,
                                        "result_file": path.relative_to(folder).as_posix()})
            print(f"[{index}] failed at {stage}: {type(error).__name__}", file=sys.stderr, flush=True)
        summary["processed"] += 1
        summary["remaining"] -= 1
        save_json(folder / "summary.json", summary)
    print(f"Outputs: {folder}", flush=True)
    return folder / "summary.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--repo-token-budget", type=int, default=None,
                        help="Serialized source budget; default 4096, or 0 for issue_only")
    parser.add_argument("--config", type=Path, help="JSON ModelSettings fields; identical file for all modes")
    parser.add_argument("--model", dest="model_name")
    for name in ("context-window", "max-output-tokens", "safety-margin", "seed", "max-retries"):
        parser.add_argument("--" + name, type=int)
    for name in ("temperature", "top-p", "timeout-seconds"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--response-format", choices=["text", "json_object", "json_schema"])
    parser.add_argument("--tokenizer", choices=["auto", "bytes"])
    parser.add_argument("--split", choices=["test", "dev", "train"], default="test")
    parser.add_argument("--limit", type=int, default=50, help="Use --limit 1 or 2 for a smoke run")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--dataset-file", type=Path, help="Existing HF Arrow file instead of a download")
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Save prompts, never call the model or score")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_bytes()) if args.config else {}
        config.update({k: getattr(args, k) for k in ModelSettings.__dataclass_fields__
                       if getattr(args, k, None) is not None})
        settings = ModelSettings(**config)
        settings.validate()
        budget = args.repo_token_budget if args.repo_token_budget is not None else (0 if args.mode == "issue_only" else 4096)
        if budget < 0 or (args.mode == "issue_only" and budget):
            raise ValueError("Source budget must be nonnegative; issue_only requires zero")
        validate_settings(args.window, args.stride)
        if args.start < 0 or args.limit <= 0:
            raise ValueError("Require start >= 0 and limit > 0")
        dataset = load_dataset_input(args.split, args.dataset_cache_dir, args.dataset_file)
        path = run_examples(dataset, args.mode, budget, args.output_dir, settings, args.start, args.limit,
                            args.split, args.window, args.stride, args.dry_run, args.resume)
        summary = json.loads(path.read_bytes())
        return int(bool(summary["failed"] or any(k.startswith("fallback_") for k in summary["status_counts"])))
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
