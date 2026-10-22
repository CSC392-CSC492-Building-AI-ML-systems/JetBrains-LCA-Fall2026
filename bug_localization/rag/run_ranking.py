"""Step 4: one label-free ranking request per saved step-3 example.

Run from the repository root: python -m bug_localization.rag.run_ranking --help
"""

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bug_localization.rag.file_ranking import complete_ranking, validate_response
from bug_localization.rag.model import ModelResponse, ModelSettings, OpenAIClient
from bug_localization.rag.prepare_repositories import RAG_ROOT, normalize_path
from bug_localization.rag.ranking_prompt import TokenCounter, pack_prompt, prompt_identity
from bug_localization.rag.retrieval import digest, make_query, project_example
from bug_localization.rag.run_retrieval import save_json


RANKING_VERSION = 1
DEFAULT_OUTPUT = RAG_ROOT / "outputs" / "llm_ranking"
COMPLETE_STATUSES = {"success", "repaired", "fallback_validation", "fallback_request", "fallback_no_evidence"}
REUSABLE_STATUSES = {"success", "repaired", "fallback_no_evidence", "dry_run"}


def contained_path(folder, name):
    path = (folder / name.replace("\\", "/")).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("Artifact path escapes run directory")
    return path


def resolve_summary(path):
    """Accept an explicit summary, a run directory, or a latest-run parent."""
    path = Path(path).resolve()
    if path.is_dir():
        if (path / "summary.json").is_file():
            return path / "summary.json"
        pointer = json.loads((path / "latest_run.json").read_bytes())
        return contained_path(path, pointer["run_directory"] + "/summary.json")
    return path


def read_checked(path):
    record = json.loads(Path(path).read_bytes())
    checksum = record.pop("payload_sha256")
    if digest(record) != checksum:
        raise ValueError("Artifact payload checksum mismatch")
    return record


def load_retrieval_run(summary_path):
    summary_path = resolve_summary(summary_path)
    summary = json.loads(summary_path.read_bytes())
    manifest = json.loads((summary_path.parent / "run_metadata.json").read_bytes())
    if (digest(manifest) != summary["run_signature"] or manifest["configuration"] != "py"
            or summary["example_indices"] != manifest["example_indices"]
            or summary["successful"] != len(summary["result_files"])
            or summary["failed"] != len(summary["failures"])
            or summary["processed"] != summary["successful"] + summary["failed"]
            or summary["processed"] + summary["remaining"] != summary["total_examples"]
            or summary["total_examples"] != len(summary["example_indices"])
            or len(set(summary["result_files"])) != len(summary["result_files"])):
        raise ValueError("Inconsistent retrieval summary/manifest")
    return summary_path, summary, manifest


def load_prediction(path, summary, manifest):
    prediction = read_checked(path)
    validate_retrieval(prediction, summary, manifest)
    return prediction


def validate_retrieval(prediction, summary, manifest):
    identity = prediction["identity"]
    metadata = identity["prepared_metadata"]
    example = project_example(identity["example_input"])
    index = prediction["example_index"]
    if (prediction["status"] != "success" or prediction["signature"] != digest(identity)
            or identity["run_signature"] != summary["run_signature"]
            or index != identity["example_index"] or index not in summary["example_indices"]
            or example != identity["example_input"] or make_query(example) != prediction["query"]
            or prediction["base_sha"] != metadata["base_sha"]
            or prediction["repository"] != metadata["repository"]
            or prediction["issue_url"] != example["issue_url"]):
        raise ValueError("Retrieval identity/provenance mismatch")
    ranking = prediction["ranking"]
    paths = [row["file"] for row in ranking]
    if (len(paths) != len(set(paths)) or set(paths) != set(metadata["candidate_paths"])
            or prediction["candidate_count"] != len(paths)
            or [row["rank"] for row in ranking] != list(range(1, len(paths) + 1))
            or any(normalize_path(p) != p for p in paths)
            or any(not math.isfinite(row["score"]) or row["score"] < 0 for row in ranking)
            or ranking != sorted(ranking, key=lambda r: (-r["score"], r["file"]))):
        raise ValueError("Retrieval ranking differs from step-1 universe")
    chunks = prediction["retrieved_chunks"]
    if (len(chunks) > prediction["requested_max_chunks"]
            or prediction["requested_max_chunks"] != manifest["settings"]["max_chunks"]
            or [c["rank"] for c in chunks] != list(range(1, len(chunks) + 1))
            or len({c["chunk_id"] for c in chunks}) != len(chunks)
            or chunks != sorted(chunks, key=lambda c: (-c["score"], c["chunk_id"]))):
        raise ValueError("Invalid retrieved evidence ordering/count")
    for chunk in chunks:
        if (chunk["path"] not in paths or type(chunk["chunk_id"]) is not int or chunk["chunk_id"] < 0
                or type(chunk["start_line"]) is not int or type(chunk["end_line"]) is not int
                or not 1 <= chunk["start_line"] <= chunk["end_line"]
                or not isinstance(chunk["text"], str) or not math.isfinite(chunk["score"])
                or chunk["score"] <= 0):
            raise ValueError("Invalid retrieved evidence")


def implementation_identity():
    names = ("run_ranking.py", "model.py", "ranking_prompt.py", "file_ranking.py",
             "retrieval.py", "prepare_repositories.py", "run_retrieval.py")
    packages = {}
    for name in ("openai", "tiktoken"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"source_hashes": {name: hashlib.sha256((RAG_ROOT / name).read_bytes()).hexdigest()
                              for name in names}, "packages": packages}


def predict(prediction, settings, counter, client=None, dry_run=False, packed=None):
    """No dataset or labels enter this function. Failed calls retain BM25."""
    packed = packed if packed is not None else pack_prompt(prediction, settings, counter)
    result = {"packing": packed, "submitted_messages": None, "model_response": None,
              "validation": None, "validated_candidate_ranking": [], "bm25_fallback": False,
              "bm25_ranking": prediction["ranking"], "ranking": prediction["ranking"]}
    if packed["skip_reason"]:
        result.update(status="fallback_no_evidence", bm25_fallback=True,
                      fallback_reason=packed["skip_reason"])
    elif dry_run:
        result.update(status="dry_run")
    else:
        if not packed["fits_budget"]:
            raise ValueError("Refusing an oversized request")
        result["submitted_messages"] = packed["messages"]
        started = time.perf_counter()
        try:
            response = (client or OpenAIClient(settings)).generate(packed["messages"])
        except Exception as error:
            response = ModelResponse(None, error={"type": type(error).__name__, "stage": "model_client"},
                                     latency_seconds=time.perf_counter() - started)
        result["model_response"] = asdict(response)
        if response.error:
            result.update(status="fallback_request", bm25_fallback=True, fallback_reason="request_failure")
        elif response.refusal or response.finish_reason not in (None, "stop"):
            result.update(status="fallback_validation", bm25_fallback=True,
                          fallback_reason="refusal_or_incomplete_response")
        else:
            validation = validate_response(response.raw_response, packed["allowed_candidate_paths"])
            result["validation"] = validation
            result["status"] = validation["status"]
            if validation["status"] == "fallback_validation":
                result.update(bm25_fallback=True, fallback_reason="unusable_response")
            else:
                result["validated_candidate_ranking"] = validation["ranking"]
                result["ranking"] = complete_ranking(validation["ranking"], prediction["ranking"])
    return result


def validate_result(result, identity, prediction):
    if result["identity"] != identity or result["signature"] != digest(identity):
        raise ValueError("Ranking identity mismatch")
    if result["bm25_ranking"] != prediction["ranking"]:
        raise ValueError("BM25 ranking was changed")
    if any(result[key] != prediction[key] for key in (
            "example_index", "issue_url", "repository", "base_sha", "candidate_count")):
        raise ValueError("Ranking provenance mismatch")
    candidate_order = result["validated_candidate_ranking"]
    if result["status"] in ("success", "repaired"):
        if (set(candidate_order) != set(result["packing"]["allowed_candidate_paths"])
                or result["bm25_fallback"]):
            raise ValueError("Incomplete LLM candidate ranking")
        expected = complete_ranking(candidate_order, prediction["ranking"])
    else:
        expected = prediction["ranking"]
    if result["ranking"] != expected:
        raise ValueError("Final ranking differs from expected complete permutation")


def run_ranking(retrieval_summary, output_root=DEFAULT_OUTPUT, settings=ModelSettings(),
                limit=1, start=0, resume=True, dry_run=False, client=None, counter=None):
    settings.validate()
    if start < 0 or (limit is not None and limit <= 0):
        raise ValueError("Require start >= 0 and positive limit")
    source_path, source_summary, source_manifest = load_retrieval_run(retrieval_summary)
    selected = [i for i in source_summary["example_indices"] if i >= start]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No selected retrieval examples")
    counter = counter or TokenCounter(settings.model_name, settings.tokenizer)
    # Validate every available selected artifact before any model request.
    predictions = {}
    for name in source_summary["result_files"]:
        path = contained_path(source_path.parent, name)
        raw = read_checked(path)
        if raw["example_index"] in selected:
            prediction = load_prediction(path, source_summary, source_manifest)
            if prediction["example_index"] in predictions:
                raise ValueError("Duplicate retrieval example")
            predictions[prediction["example_index"]] = prediction
    inputs = {str(i): digest(p) for i, p in predictions.items()}
    manifest = {
        "ranking_version": RANKING_VERSION, "configuration": "py", "split": source_manifest["split"],
        "dataset": source_manifest["dataset"], "dataset_fingerprint": source_manifest["dataset_fingerprint"],
        "dataset_total_examples": source_manifest["dataset_total_examples"], "example_indices": selected,
        "retrieval_run_signature": source_summary["run_signature"], "retrieval_inputs": inputs,
        "retrieval_manifest": source_manifest, "model": settings.identity(), "prompt": prompt_identity(),
        "token_counter": counter.identity, "dry_run": dry_run, **implementation_identity(),
    }
    signature = digest(manifest)
    folder = Path(output_root).resolve() / signature
    save_json(folder / "run_metadata.json", manifest)
    save_json(folder.parent / "latest_run.json", {"run_directory": signature})
    summary = {"baseline": "BM25 + LLM RAG", "configuration": "py", "split": manifest["split"],
               "run_signature": signature, "total_examples": len(selected), "example_indices": selected,
               "dataset_total_examples": manifest["dataset_total_examples"], "processed": 0,
               "remaining": len(selected), "successful": 0, "failed": 0, "reused": 0,
               "result_files": [], "failures": [], "status_counts": {}, "dry_run": dry_run}
    save_json(folder / "summary.json", summary)
    for index in selected:
        path = folder / "examples" / f"{index:03d}.json"
        started = time.perf_counter()
        try:
            if index not in predictions:
                raise ValueError("Step-3 prediction unavailable; no valid BM25 fallback exists")
            prediction = predictions[index]
            identity = {"run_signature": signature, "example_index": index,
                        "retrieval_sha256": inputs[str(index)]}
            result = None
            if resume:
                try:
                    cached = read_checked(path)
                    validate_result(cached, identity, prediction)
                    if cached["status"] in REUSABLE_STATUSES:
                        result = cached
                        summary["reused"] += 1
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if result is None:
                base = {"identity": identity, "signature": digest(identity), "example_index": index,
                        "issue_url": prediction["issue_url"], "repository": prediction["repository"],
                        "base_sha": prediction["base_sha"], "candidate_count": prediction["candidate_count"],
                        "prepared_snapshot": prediction["prepared_snapshot"],
                        "snapshot_metadata": prediction["identity"]["prepared_metadata"],
                        "model_settings": settings.identity(),
                        "retrieval_result": {"sha256": inputs[str(index)],
                                             "signature": prediction["signature"],
                                             "run_signature": source_summary["run_signature"]},
                        "retrieval_prediction": prediction}
                packed = pack_prompt(prediction, settings, counter)
                running = {**base, "status": "running", "packing": packed,
                           "request_state": "not_sent" if dry_run or packed["skip_reason"] else "pending_or_in_flight"}
                save_json(path, {**running, "payload_sha256": digest(running)})
                result = {**base, **predict(prediction, settings, counter, client, dry_run, packed),
                          "elapsed_seconds": time.perf_counter() - started}
                validate_result(result, identity, prediction)
                save_json(path, {**result, "payload_sha256": digest(result)})
            status = result["status"]
            summary["result_files"].append(path.relative_to(folder).as_posix())
            # 'successful' counts completed pipeline artifacts, including BM25
            # fallbacks. Model success is separately visible in status_counts.
            summary["successful"] += 1
            summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
            print(f"[{index}] {status}; packed_files={len(result['packing']['allowed_candidate_paths'])}", flush=True)
        except Exception as error:
            # Exception text may embed SDK requests/credentials. Save only type and stage.
            failure = {"status": "failed", "example_index": index, "error_type": type(error).__name__,
                       "stage": "ranking_pipeline", "elapsed_seconds": time.perf_counter() - started}
            save_json(path, {**failure, "payload_sha256": digest(failure)})
            summary["failed"] += 1
            summary["failures"].append(failure)
            print(f"[{index}] failed: {type(error).__name__}", file=sys.stderr, flush=True)
        summary["processed"] += 1
        summary["remaining"] -= 1
        save_json(folder / "summary.json", summary)
    print(f"Outputs: {folder}", flush=True)
    return folder / "summary.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("retrieval", type=Path, help="Step-3 summary.json, run folder, or latest-run parent")
    parser.add_argument("--config", type=Path, help="JSON object of ModelSettings fields; no credentials")
    parser.add_argument("--backend", choices=["openai"])
    parser.add_argument("--model", dest="model_name")
    for flag in ("context-window", "max-output-tokens", "max-retries", "safety-margin"):
        parser.add_argument("--" + flag, type=int)
    for flag in ("temperature", "top-p", "timeout-seconds"):
        parser.add_argument("--" + flag, type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--response-format", choices=["json_schema", "json_object", "text"])
    parser.add_argument("--tokenizer", choices=["auto", "bytes"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="Minimum original dataset index")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="All selected step-3 examples from --start")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--tfidf-summary", type=Path, help="Existing TF-IDF summary for matched comparison")
    parser.add_argument("--dataset-file", type=Path, help="Existing Arrow file, only opened during evaluation")
    parser.add_argument("--dataset-cache-dir", type=Path)
    args = parser.parse_args(argv)
    if args.dry_run and args.evaluate:
        parser.error("Dry-run prompts are not model predictions; evaluate a real run")
    try:
        config = json.loads(args.config.read_bytes()) if args.config else {}
        config.update({key: getattr(args, key) for key in ModelSettings.__dataclass_fields__
                       if getattr(args, key, None) is not None})
        settings = ModelSettings(**config)
        path = run_ranking(args.retrieval, args.output_dir, settings,
                           None if args.all else args.limit, args.start, args.resume, args.dry_run)
        summary = json.loads(path.read_bytes())
        if args.evaluate:
            from bug_localization.rag.evaluate_ranking import evaluate_run
            from bug_localization.rag.run_retrieval import load_dataset_input
            dataset = load_dataset_input(summary["split"], args.dataset_cache_dir, args.dataset_file)
            metrics = evaluate_run(path, dataset, args.tfidf_summary)
            print(json.dumps(metrics, indent=2))
        return int(bool(summary["failed"] or summary["status_counts"].get("fallback_request")
                        or summary["status_counts"].get("fallback_validation")))
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
