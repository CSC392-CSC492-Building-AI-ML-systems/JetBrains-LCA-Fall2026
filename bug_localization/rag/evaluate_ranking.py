"""Post-prediction evaluation; the only step-4 module permitted to read labels."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bug_localization.baseline_eval.evaluate_tfidf import aggregate, score_example
from bug_localization.baseline_eval.run_all_tfidf import get_labels
from bug_localization.rag.evaluate_retrieval import candidate_metrics
from bug_localization.rag.retrieval import digest, project_example
from bug_localization.rag.run_ranking import (
    COMPLETE_STATUSES, contained_path, read_checked, resolve_summary, validate_result, validate_retrieval,
)
from bug_localization.rag.run_retrieval import load_dataset_input, save_json


def _tfidf_results(summary_path, manifest):
    if summary_path is None:
        return None
    path = resolve_summary(summary_path)
    summary = json.loads(path.read_bytes())
    metadata = json.loads((path.parent / "run_metadata.json").read_bytes())
    if (summary["configuration"] != manifest["configuration"] or summary["split"] != manifest["split"]
            or metadata["dataset_fingerprint"] != manifest["dataset_fingerprint"]
            or len(summary["result_files"]) != summary["successful"]):
        raise ValueError("TF-IDF split/fingerprint/summary differs from the RAG run")
    results = {}
    for name in summary["result_files"]:
        row = json.loads(contained_path(path.parent, name).read_bytes())
        if row["status"] != "success" or row["example_index"] in results:
            raise ValueError("Failed or duplicate TF-IDF prediction")
        results[row["example_index"]] = row
    return results


def evaluate_run(summary_path, dataset, tfidf_summary=None):
    path = resolve_summary(summary_path)
    folder = path.parent
    summary = json.loads(path.read_bytes())
    manifest = json.loads((folder / "run_metadata.json").read_bytes())
    if (digest(manifest) != summary["run_signature"] or manifest["dry_run"]
            or manifest["dataset_total_examples"] != len(dataset)
            or manifest["dataset_fingerprint"] != getattr(dataset, "_fingerprint", None)
            or summary["example_indices"] != manifest["example_indices"]
            or len(summary["result_files"]) != summary["successful"]
            or len(summary["failures"]) != summary["failed"]
            or summary["successful"] + summary["failed"] != summary["processed"]
            or summary["processed"] + summary["remaining"] != summary["total_examples"]):
        raise ValueError("Invalid/incomplete run identity, dataset mismatch, or dry-run evaluation")
    tfidf = _tfidf_results(tfidf_summary, manifest)
    rows = {"BM25": [], "BM25 + LLM RAG": []}
    if tfidf is not None:
        rows["TF-IDF"] = []
    details, responses, seen = [], [], set()
    for name in summary["result_files"]:
        result = read_checked(contained_path(folder, name))
        index = result["example_index"]
        if index in seen or index not in manifest["example_indices"] or result["status"] not in COMPLETE_STATUSES:
            raise ValueError("Duplicate, unselected, or incomplete prediction")
        seen.add(index)
        prediction = result["retrieval_prediction"]
        expected = {"run_signature": summary["run_signature"], "example_index": index,
                    "retrieval_sha256": manifest["retrieval_inputs"][str(index)]}
        if digest(prediction) != expected["retrieval_sha256"]:
            raise ValueError("Saved retrieval input differs from manifest")
        source_manifest = manifest["retrieval_manifest"]
        source_summary = {"run_signature": manifest["retrieval_run_signature"],
                          "example_indices": source_manifest["example_indices"]}
        validate_retrieval(prediction, source_summary, source_manifest)
        validate_result(result, expected, prediction)
        # Labels and full dataset rows are accessed only now, after saved predictions.
        row = dataset[index]
        if project_example(row) != prediction["identity"]["example_input"]:
            raise ValueError("Evaluation example differs from prediction inputs")
        labels = get_labels(row["changed_files"])
        common = {"status": "success", "example_index": index, "issue_url": result["issue_url"],
                  "repository": result["repository"], "base_sha": result["base_sha"],
                  "candidate_count": result["candidate_count"], "correct_files": labels}
        for method, ranking in (("BM25", result["bm25_ranking"]), ("BM25 + LLM RAG", result["ranking"])):
            adapted = {**common, "ranking": ranking}
            rows[method].append(score_example(adapted))
            save_json(folder / "evaluation" / ("bm25" if method == "BM25" else "rag")
                      / f"{index:03d}.json", adapted)
        if tfidf is not None:
            baseline = tfidf.get(index)
            if (baseline is None or any(baseline[k] != common[k] for k in (
                    "issue_url", "repository", "base_sha", "candidate_count"))
                    or set(baseline["correct_files"]) != set(labels)
                    or {r["file"] for r in baseline["ranking"]} != {r["file"] for r in result["ranking"]}):
                raise ValueError(f"TF-IDF example {index} is missing or differs in snapshot/labels/candidate universe")
            rows["TF-IDF"].append(score_example(baseline))
        before = candidate_metrics(prediction, labels)
        after = candidate_metrics({**prediction, "retrieved_chunks": [c for c in prediction["retrieved_chunks"]
                                  if c["chunk_id"] in result["packing"]["included_chunk_ids"]]}, labels)
        details.append({"example_index": index, "status": result["status"],
                        "before_packing": before, "after_packing": after})
        if result["model_response"] is not None:
            responses.append(result["model_response"])
    statuses = Counter(detail["status"] for detail in details)
    usage = Counter()
    usage_examples = Counter()
    for response in responses:
        for key, value in response["usage"].items():
            if type(value) in (int, float):
                usage[key] += value
                usage_examples[key] += 1
    def recall(phase):
        applicable = [d[phase]["candidate_file_recall"] for d in details
                      if d[phase]["candidate_file_recall"] is not None]
        return {"macro": mean(applicable) if applicable else None, "applicable_examples": len(applicable),
                "zero_eligible_gold_examples": len(details) - len(applicable)}
    output = {
        "evaluation_version": 1, "configuration": manifest["configuration"], "split": manifest["split"],
        "scope": "Completed pipeline predictions including every recorded BM25 fallback",
        "total_examples": summary["total_examples"], "evaluated_examples": len(details),
        "coverage": len(details) / summary["total_examples"], "pipeline_failures": summary["failed"],
        "remaining_examples": summary["remaining"], "failures": summary["failures"],
        "file_ranking_metrics": {method: aggregate(values) if values else None for method, values in rows.items()},
        "comparison": "Same examples and candidate universes" if tfidf is not None else
                      "BM25/RAG only; supply --tfidf-summary for the three-way comparison",
        "retrieval_recall_before_packing": recall("before_packing"),
        "retrieval_recall_after_packing": recall("after_packing"),
        "ground_truth_outside_candidates_total": sum(len(d["before_packing"]["ground_truth_outside_candidates"])
                                                     for d in details),
        "successful_model_responses": statuses["success"] + statuses["repaired"],
        "responses_requiring_repairs": statuses["repaired"],
        "model_request_failures": statuses["fallback_request"],
        "unusable_model_responses": statuses["fallback_validation"],
        "bm25_fallbacks": sum(statuses[s] for s in statuses if s.startswith("fallback_")),
        "status_counts": dict(statuses), "usage_totals": dict(usage),
        "usage_reporting_examples": dict(usage_examples),
        "request_latency_seconds_total": sum(r["latency_seconds"] for r in responses),
        "request_latency_seconds_mean": mean(r["latency_seconds"] for r in responses) if responses else None,
        "transport_retries": sum(max(0, len(r["attempts"]) - 1) for r in responses),
        "definitions": {"ranking_metrics": "Unchanged evaluate_tfidf.score_example/aggregate",
                        "recall": "Distinct gold candidate files in evidence / all gold candidate files; zero eligible -> null",
                        "outside_candidate_gold": "Retained as misses in file-ranking metric denominators",
                        "ordinal_scores": "RAG scores encode rank only; BM25 scores remain separately preserved",
                        "usage": "Available numeric usage fields only; failed attempts may have unreported billable usage"},
    }
    save_json(folder / "evaluation" / "metrics.json", output)
    save_json(folder / "evaluation" / "per_example_metrics.json", [
        {**d, "file_ranking_metrics": {method: values[i] for method, values in rows.items()}}
        for i, d in enumerate(details)])
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--tfidf-summary", type=Path)
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--dataset-cache-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        path = resolve_summary(args.summary)
        summary = json.loads(path.read_bytes())
        dataset = load_dataset_input(summary["split"], args.dataset_cache_dir, args.dataset_file)
        print(json.dumps(evaluate_run(path, dataset, args.tfidf_summary), indent=2))
        return int(bool(summary["failed"]))
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
