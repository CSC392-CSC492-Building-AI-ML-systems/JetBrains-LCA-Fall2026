"""Step 5: sequential, resumable Python test benchmark. No live calls in dry run."""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from bug_localization.rag import bm25_index, prepare_repositories as preparation
from bug_localization.rag import run_ranking, run_retrieval
from bug_localization.rag.retrieval_tokenizer import tokenizer_identity
from bug_localization.rag.model import ModelSettings
from bug_localization.rag.ranking_prompt import TokenCounter, pack_prompt, prompt_identity
from bug_localization.rag.retrieval import digest, project_example
from bug_localization.rag.run_retrieval import save_json


DEFAULT_OUTPUT = preparation.RAG_ROOT.parent / "outputs" / "rag_python50"
DEFAULT_CONFIG = preparation.RAG_ROOT / "benchmark_config.json"
LABEL_POLICY = "Raw dataset changed_files; unreachable labels count as misses"


def source_hashes():
    paths = list(preparation.RAG_ROOT.glob("*.py"))
    paths.extend((preparation.RAG_ROOT.parent / "baseline_eval" / name)
                 for name in ("evaluate_tfidf.py", "run_all_tfidf.py"))
    return {str(p.relative_to(preparation.RAG_ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def result_identity(signature, index, prediction):
    return {"run_signature": signature, "example_index": index,
            "retrieval_sha256": digest(prediction)}


def validate_saved(result, manifest, signature):
    index = result["example_index"]
    prediction = result["retrieval_prediction"]
    example = manifest["examples"][str(index)]
    source = result["retrieval_source_manifest"]
    if (prediction["identity"]["example_input"] != example["input"]
            or result["example_id"] != example["example_id"]
            or source["dataset_fingerprint"] != manifest["dataset_fingerprint"]
            or source["dataset_total_examples"] != manifest["dataset_total_examples"]
            or source["dataset"] != manifest["dataset"] or source["configuration"] != "py"
            or source["split"] != "test" or source["tokenizer"] != manifest["retrieval_tokenizer"]
            or source["bm25_scoring_version"] != manifest["bm25_scoring_version"]
            or source["settings"] != manifest["retrieval_settings"]
            or source["source_hashes"] != manifest["retrieval_source_hashes"]
            or source["example_indices"] != [index]
            or result["model_settings"] != manifest["model"]
            or result["packing"] != pack_prompt(prediction, ModelSettings(**{
                k: manifest["model"][k] for k in ModelSettings.__dataclass_fields__}),
                TokenCounter(manifest["model"]["model_name"], manifest["model"]["tokenizer"]))):
        raise ValueError("Benchmark inputs/configuration differ from saved prediction")
    run_ranking.validate_retrieval(prediction, {"run_signature": digest(source),
                                  "example_indices": [index]}, source)
    run_ranking.validate_result(result, result_identity(signature, index, prediction), prediction)


def run_benchmark(dataset, output_root=DEFAULT_OUTPUT, indices=None, settings=None,
                  retrieval_settings=run_retrieval.Settings(), dry_run=False, resume=True,
                  retry_fallbacks=False, prepared_cache=preparation.DEFAULT_CACHE,
                  index_cache=bm25_index.DEFAULT_CACHE, repository_cache_dir=None, client=None):
    settings = settings or ModelSettings(**json.loads(DEFAULT_CONFIG.read_bytes()))
    settings.validate()
    retrieval_settings.validate()
    if len(dataset) != 50:
        raise ValueError(f"Expected 50 Python test examples, received {len(dataset)}")
    indices = list(range(50) if indices is None else indices)
    if (not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < 50 for i in indices)):
        raise ValueError("Require distinct original test indices in [0, 49]")
    projected = run_retrieval.prediction_columns(dataset)
    examples = {str(i): {"input": project_example(projected[i]),
                         "example_id": digest(project_example(projected[i]))[:16]} for i in indices}
    counter = TokenCounter(settings.model_name, settings.tokenizer)
    manifest = {
        "benchmark_version": 1, "dataset": preparation.DATASET, "configuration": "py", "split": "test",
        "dataset_total_examples": len(dataset), "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "example_indices": indices, "examples": examples, "dry_run": dry_run,
        "retrieval_settings": asdict(retrieval_settings),
        "retrieval_tokenizer": tokenizer_identity(), "bm25_scoring_version": bm25_index.SCORING_VERSION,
        "query_policy": "nonblank-issue_title-blank-line-issue_body/v1",
        "file_scoring": "maximum-over-all-indexed-chunks; ties=path/v1",
        "retrieval_source_hashes": run_retrieval.implementation_hashes(),
        "candidate_policy": preparation.CANDIDATE_POLICY, "label_policy": LABEL_POLICY,
        "evaluation_policy": "Unchanged evaluate_tfidf.score_example/aggregate; failures excluded and disclosed",
        "model": settings.identity(), "prompt": prompt_identity(), "token_counter": counter.identity,
        "ranking_implementation": run_ranking.implementation_identity(), "source_hashes": source_hashes(),
    }
    signature = digest(manifest)
    folder = Path(output_root).resolve() / signature
    print(f"Outputs: {folder}\nModel: {settings.model_name}; examples: {len(indices)}; "
          f"logical request limit: {0 if dry_run else len(indices)}; "
          f"transport retries/request: {settings.max_retries}; dry_run={dry_run}", flush=True)
    save_json(folder / "run_metadata.json", manifest)
    save_json(folder.parent / "latest_run.json", {"run_directory": signature})
    summary = {"baseline": "BM25 + LLM RAG", "configuration": "py", "split": "test",
               "run_signature": signature, "example_indices": indices, "dataset_total_examples": 50,
               "total_examples": len(indices), "processed": 0, "successful": 0, "failed": 0,
               "remaining": len(indices), "successful_model_rankings": 0, "completed_bm25_fallbacks": 0,
               "dry_run_prompts": 0, "reused": 0, "dry_run": dry_run,
               "result_files": [], "failures": [], "status_counts": {}}
    save_json(folder / "summary.json", summary)
    for i in indices:
        started = time.perf_counter()
        path = folder / "examples" / f"{i:03d}_{examples[str(i)]['example_id']}.json"
        result = None
        previous_attempts = []
        try:
            if resume:
                try:
                    cached = run_ranking.read_checked(path)
                    validate_saved(cached, manifest, signature)
                    reusable = cached["status"] in run_ranking.COMPLETE_STATUSES | {"dry_run"}
                    previous_attempts = cached.get("previous_attempts", [])
                    if reusable and retry_fallbacks and cached["bm25_fallback"]:
                        archive = folder / "attempts" / f"{i:03d}_{digest(cached)}.json"
                        save_json(archive, {**cached, "payload_sha256": digest(cached)})
                        previous_attempts = [*previous_attempts, archive.relative_to(folder).as_posix()]
                    if reusable and not (retry_fallbacks and cached["bm25_fallback"]):
                        result = cached
                        summary["reused"] += 1
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if result is None:
                # Reuse step 3 intact, with original indices and label-free projection.
                source_path = run_retrieval.run_examples(
                    dataset, folder / "retrieval", [i], retrieval_settings, "test",
                    prepared_cache, index_cache, repository_cache_dir, resume)
                _, source_summary, source_manifest = run_ranking.load_retrieval_run(source_path)
                if source_summary["failed"]:
                    raise RuntimeError("Retrieval failed; no valid full ranking")
                prediction = run_ranking.load_prediction(run_ranking.contained_path(
                    source_path.parent, source_summary["result_files"][0]), source_summary, source_manifest)
                identity = result_identity(signature, i, prediction)
                base = {"identity": identity, "signature": digest(identity), "example_index": i,
                        "example_id": examples[str(i)]["example_id"],
                        **{k: prediction[k] for k in ("issue_url", "repository", "base_sha", "candidate_count")},
                        "retrieval_prediction": prediction, "retrieval_source_manifest": source_manifest,
                        "retrieval_summary_path": str(source_path.resolve()),
                        "previous_attempts": previous_attempts,
                        "model_settings": settings.identity()}
                packed = pack_prompt(prediction, settings, counter)
                running = {**base, "status": "running", "packing": packed,
                           "request_state": "not_sent" if dry_run or packed["skip_reason"] else "pending_or_in_flight"}
                save_json(path, {**running, "payload_sha256": digest(running)})
                predicted = run_ranking.predict(prediction, settings, counter, client, dry_run, packed)
                if dry_run:
                    # Step 4 marks empty evidence as fallback even in dry runs.
                    # Here every dry result is explicitly non-predictive.
                    predicted.update(status="dry_run", bm25_fallback=False)
                result = {**base, **predicted, "elapsed_seconds": time.perf_counter() - started}
                validate_saved(result, manifest, signature)
                save_json(path, {**result, "payload_sha256": digest(result)})
            status = result["status"]
            summary["successful"] += 1
            summary["successful_model_rankings"] += int(status in ("success", "repaired"))
            summary["completed_bm25_fallbacks"] += int(result["bm25_fallback"])
            summary["dry_run_prompts"] += int(status == "dry_run")
            summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
            summary["result_files"].append(path.relative_to(folder).as_posix())
            print(f"[{i}] {status}", flush=True)
        except Exception as error:
            failure = {"status": "failed", "example_index": i, "example_id": examples[str(i)]["example_id"],
                       "error_type": type(error).__name__, "stage": "benchmark_pipeline",
                       "elapsed_seconds": time.perf_counter() - started}
            save_json(path, {**failure, "payload_sha256": digest(failure)})
            summary["failed"] += 1
            summary["failures"].append(failure)
            print(f"[{i}] failed: {type(error).__name__}", file=sys.stderr, flush=True)
        summary["processed"] += 1
        summary["remaining"] -= 1
        save_json(folder / "summary.json", summary)
    print(f"Outputs: {folder}", flush=True)
    return folder / "summary.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", dest="model_name")
    parser.add_argument("--backend", choices=["openai"])
    for key in ("context-window", "max-output-tokens", "seed", "max-retries", "safety-margin"):
        parser.add_argument("--" + key, type=int)
    for key in ("temperature", "top-p", "timeout-seconds"):
        parser.add_argument("--" + key, type=float)
    parser.add_argument("--response-format", choices=["text", "json_object", "json_schema"])
    parser.add_argument("--tokenizer", choices=["auto", "bytes"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-fallbacks", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepared-cache-dir", type=Path, default=preparation.DEFAULT_CACHE)
    parser.add_argument("--index-cache-dir", type=Path, default=bm25_index.DEFAULT_CACHE)
    parser.add_argument("--repository-cache-dir", type=Path)
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--tfidf-summary", type=Path, default=preparation.RAG_ROOT.parent / "outputs" / "tfidf_python50")
    parser.add_argument("--evaluate-only", type=Path, help="Saved run folder/summary; never runs inference")
    args = parser.parse_args(argv)
    try:
        dataset = run_retrieval.load_dataset_input("test", args.dataset_cache_dir, args.dataset_file)
        if args.evaluate_only:
            from bug_localization.rag.evaluate_benchmark import evaluate
            metrics = evaluate(args.evaluate_only, dataset, args.tfidf_summary)
            print(json.dumps(metrics, indent=2))
            return int(bool(metrics["pipeline_failures"] or metrics["remaining_examples"]))
        config = json.loads(args.config.read_bytes())
        config.update({k: getattr(args, k) for k in ModelSettings.__dataclass_fields__
                       if getattr(args, k, None) is not None})
        if not 0 <= args.start < 50 or args.limit <= 0:
            raise ValueError("Require 0 <= start < 50 and positive limit")
        if args.all and args.start != 0:
            raise ValueError("--all requires --start 0 for the complete 50-example split")
        indices = range(50) if args.all else range(args.start, min(50, args.start + args.limit))
        path = run_benchmark(dataset, args.output_dir, indices, ModelSettings(**config),
                             dry_run=args.dry_run, resume=args.resume, retry_fallbacks=args.retry_fallbacks,
                             prepared_cache=args.prepared_cache_dir, index_cache=args.index_cache_dir,
                             repository_cache_dir=args.repository_cache_dir)
        if not args.dry_run:
            from bug_localization.rag.evaluate_benchmark import evaluate
            metrics = evaluate(path, dataset, args.tfidf_summary)
            print(json.dumps(metrics, indent=2))
        return int(bool(json.loads(path.read_bytes())["failed"]))
    except Exception as error:
        print(f"FAILED: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
