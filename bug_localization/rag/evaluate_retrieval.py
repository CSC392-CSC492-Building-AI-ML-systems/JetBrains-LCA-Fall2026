"""Evaluate saved BM25 retrieval predictions; this is the only label-reading step.

Reuse the TF-IDF evaluator's score_example/aggregate without changing its metric
definitions. Candidate recall separately uses gold paths within the candidates.
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bug_localization.baseline_eval.evaluate_tfidf import aggregate, score_example
from bug_localization.baseline_eval.run_all_tfidf import get_labels
from bug_localization.rag.retrieval import digest, make_query, project_example
from bug_localization.rag.run_retrieval import load_dataset_input, save_json


EVALUATION_VERSION = 1


def candidate_metrics(prediction, labels):
    """Use exact path equality, as in the existing file-ranking evaluator.

    There is no additional gold test-file filter or case/slash normalization.
    With zero eligible gold files, recall and both hit indicators are undefined;
    report null and exclude them from applicable-example macro denominators.
    """
    candidates = {row["file"] for row in prediction["ranking"]}
    retrieved = {chunk["path"] for chunk in prediction["retrieved_chunks"]}
    if not retrieved.issubset(candidates):
        raise ValueError("Retrieved file is outside the candidate universe")
    eligible = set(labels) & candidates
    hits = eligible & retrieved
    return {
        "example_index": prediction["example_index"],
        "eligible_ground_truth_files": sorted(eligible),
        "ground_truth_outside_candidates": [path for path in labels if path not in candidates],
        "eligible_ground_truth_count": len(eligible), "retrieved_eligible_count": len(hits),
        "candidate_file_recall": len(hits) / len(eligible) if eligible else None,
        "at_least_one_eligible_retrieved": int(bool(hits)) if eligible else None,
        "all_eligible_retrieved": int(hits == eligible) if eligible else None,
        "retrieved_chunk_count": len(prediction["retrieved_chunks"]),
        "requested_max_chunks": prediction["requested_max_chunks"],
        "distinct_retrieved_files": len(retrieved),
    }


def _result_path(folder, name):
    path = (folder / name.replace("\\", "/")).resolve()
    if not path.is_relative_to(folder):
        raise ValueError("Result path escapes the run directory")
    return path


def evaluate_run(summary_path, dataset):
    """Read labels only for already-produced successful predictions.

    Return metrics on valid successful examples, with all prediction failures,
    evaluation failures, and remaining examples explicitly counted alongside.
    Save labeled evaluator-compatible records separately under evaluation/.
    """
    summary_path = Path(summary_path).resolve()
    folder = summary_path.parent
    summary = json.loads(summary_path.read_bytes())
    manifest = json.loads((folder / "run_metadata.json").read_bytes())
    if (digest(manifest) != summary["run_signature"]
            or len(dataset) != manifest["dataset_total_examples"]
            or len(summary["result_files"]) != summary["successful"]
            or len(summary["failures"]) != summary["failed"]
            or summary["processed"] != summary["successful"] + summary["failed"]
            or summary["processed"] + summary["remaining"] != summary["total_examples"]
            or summary["example_indices"] != manifest["example_indices"]):
        raise ValueError("Inconsistent run summary, manifest, or dataset size")
    fingerprint = getattr(dataset, "_fingerprint", None)
    if fingerprint != manifest["dataset_fingerprint"]:
        raise ValueError("Evaluation dataset fingerprint differs from the prediction run")
    labeled_dir = folder / "evaluation"
    rows, recall_rows, labeled_files, evaluation_failures = [], [], [], []
    seen_paths, seen_indices = set(), set()
    for name in summary["result_files"]:
        try:
            path = _result_path(folder, name)
            if path in seen_paths:
                raise ValueError("Duplicate result path")
            seen_paths.add(path)
            prediction = json.loads(path.read_bytes())
            checksum = prediction.pop("payload_sha256")
            identity = prediction["identity"]
            example_index = prediction["example_index"]
            if (checksum != digest(prediction) or prediction["status"] != "success"
                    or prediction["signature"] != digest(identity)
                    or identity["run_signature"] != summary["run_signature"]
                    or identity["example_index"] != example_index
                    or example_index not in summary["example_indices"] or example_index in seen_indices):
                raise ValueError("Prediction checksum, identity, or example index mismatch")
            seen_indices.add(example_index)
            # Evaluation begins here, after the runner has persisted predictions.
            row = dataset[example_index]
            example = project_example(row)
            if (example != identity["example_input"] or make_query(example) != prediction["query"]
                    or prediction["base_sha"] != identity["prepared_metadata"]["base_sha"]
                    or prediction["requested_max_chunks"] != manifest["settings"]["max_chunks"]):
                raise ValueError("Evaluation example does not match the prediction inputs")
            labels = get_labels(row["changed_files"])
            adapted = {
                "status": "success", "example_index": example_index,
                "issue_url": prediction["issue_url"], "repository": prediction["repository"],
                "base_sha": prediction["base_sha"], "candidate_count": prediction["candidate_count"],
                "ranking": prediction["ranking"], "correct_files": labels,
                "unranked_correct_files": [label for label in labels
                                           if label not in {r["file"] for r in prediction["ranking"]}],
            }
            ranking_metrics = score_example(adapted)
            recall = candidate_metrics(prediction, labels)
            label_path = labeled_dir / "examples" / f"{example_index:03d}.json"
            save_json(label_path, adapted)
            labeled_files.append(label_path.relative_to(labeled_dir).as_posix())
            rows.append(ranking_metrics)
            recall_rows.append(recall)
        except Exception as error:
            evaluation_failures.append({"result_file": name, "error": f"{type(error).__name__}: {error}"})
    eligible_rows = [row for row in recall_rows if row["eligible_ground_truth_count"]]
    selected_total = summary["total_examples"]
    output = {
        "evaluation_version": EVALUATION_VERSION, "baseline": summary["baseline"],
        "configuration": summary["configuration"], "split": summary["split"],
        "scope": "Valid successful predictions only; see coverage and failures",
        "total_examples": selected_total, "dataset_total_examples": len(dataset),
        "prediction_successes": summary["successful"], "prediction_failures": summary["failed"],
        "evaluation_failures": len(evaluation_failures), "remaining_examples": summary["remaining"],
        "evaluated_examples": len(rows), "coverage": len(rows) / selected_total,
        "requested_max_chunks": manifest["settings"]["max_chunks"],
        "candidate_recall_applicable_examples": len(eligible_rows),
        "examples_without_eligible_ground_truth": len(recall_rows) - len(eligible_rows),
        "candidate_file_recall_macro": mean(r["candidate_file_recall"] for r in eligible_rows) if eligible_rows else None,
        "fraction_at_least_one_eligible_retrieved": mean(r["at_least_one_eligible_retrieved"] for r in eligible_rows) if eligible_rows else None,
        "fraction_all_eligible_retrieved": mean(r["all_eligible_retrieved"] for r in eligible_rows) if eligible_rows else None,
        "distinct_retrieved_files_mean": mean(r["distinct_retrieved_files"] for r in recall_rows) if recall_rows else None,
        "ground_truth_outside_candidates_total": sum(len(r["ground_truth_outside_candidates"]) for r in recall_rows),
        "file_ranking_metrics": aggregate(rows) if rows else None,
        "failures": {"prediction": summary["failures"], "evaluation": evaluation_failures},
        "definitions": {
            "candidate_file_recall": "Retrieved gold candidate paths / all gold candidate paths; distinct files, not chunks",
            "zero_eligible_gold": "Per-example recall/hit fields are null; excluded from applicable-example macro denominators",
            "paths": "Exact path equality and raw get_labels, unchanged from evaluate_tfidf; no extra normalization/filtering",
            "file_ranking": "Unchanged evaluate_tfidf.score_example/aggregate; outside-candidate gold remains in metric denominators",
        },
    }
    save_json(labeled_dir / "metrics.json", output)
    save_json(labeled_dir / "per_example_metrics.json", [
        {**ranking, "candidate_retrieval": recall} for ranking, recall in zip(rows, recall_rows)])
    # Also allow the original standalone evaluator to consume the full ranking
    # artifacts for Python test runs (pass --allow-partial for smoke subsets).
    save_json(labeled_dir / "summary.json", {
        **summary, "successful": len(rows), "failed": summary["failed"] + len(evaluation_failures),
        "result_files": labeled_files,
    })
    return output


def print_metrics(metrics):
    print(f"Evaluation: {metrics['evaluated_examples']}/{metrics['total_examples']} examples; "
          f"prediction_failures={metrics['prediction_failures']} evaluation_failures={metrics['evaluation_failures']} "
          f"remaining={metrics['remaining_examples']}")
    for key in ("candidate_file_recall_macro", "fraction_at_least_one_eligible_retrieved",
                "fraction_all_eligible_retrieved", "distinct_retrieved_files_mean"):
        value = metrics[key]
        print(f"{key}: {value:.6f}" if value is not None else f"{key}: N/A")
    if metrics["file_ranking_metrics"]:
        for key, value in metrics["file_ranking_metrics"].items():
            print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Saved retrieval run summary.json")
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--dataset-file", type=Path, help="Existing Arrow file for the run's split (offline)")
    args = parser.parse_args(argv)
    try:
        summary = json.loads(args.summary.read_bytes())
        dataset = load_dataset_input(summary["split"], args.dataset_cache_dir, args.dataset_file)
        metrics = evaluate_run(args.summary, dataset)
        print_metrics(metrics)
        print(f"Saved: {args.summary.resolve().parent / 'evaluation'}")
        return int(bool(metrics["prediction_failures"] or metrics["evaluation_failures"]))
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
