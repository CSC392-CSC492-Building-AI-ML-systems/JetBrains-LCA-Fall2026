r"""Evaluate the complete Python test run from run_all_tfidf.py.

This file lives in bug_localization/baseline_eval. Run from lca-baselines:
  .\.venv\Scripts\python.exe .\bug_localization\baseline_eval\evaluate_tfidf.py
Defaults to the runner's latest_run.json pointer. An explicit summary path
overrides it. Use --allow-partial only to intentionally evaluate a partial run.
Uses only the Python standard library; no rerunning models or downloads.

AP uses the saved ranked order (including its existing tie order):
  AP = sum(precision at each relevant rank) / number of relevant files.
Correct files outside the candidate snapshot contribute zero; they remain
in ground-truth denominators. Null ranks represent unretrieved correct files.
The default requires all 50 examples and refuses an incomplete run.
F1_at_k_macro uses k=1 for single-file and k=2 for multiple-file cases.
It is explicitly separate from the paper's reported F1: the archived notebook
does not consistently apply k in its F1 helper. Exact paper-score reproduction
has not been verified. Compare methods on identical examples and definitions.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def score_example(result):
    if result.get("status") != "success":
        raise ValueError("Listed result is not a successful run")
    labels = result["correct_files"]
    if not isinstance(labels, list) or not labels or not all(isinstance(x, str) for x in labels):
        raise ValueError("Invalid correct_files")
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate ground-truth paths")
    gold = set(labels)
    ranking = result["ranking"]
    if not ranking:
        raise ValueError("Empty ranking")
    if [row["rank"] for row in ranking] != list(range(1, len(ranking) + 1)):
        raise ValueError("Ranking is not in consecutive saved rank order")
    files = [row["file"] for row in ranking]
    if len(set(files)) != len(files):
        raise ValueError("Duplicate candidate paths")
    if result.get("candidate_count", len(files)) != len(files):
        raise ValueError("Incomplete ranking")
    scores = [float(row["score"]) for row in ranking]
    if not all(math.isfinite(x) for x in scores):
        raise ValueError("Non-finite scores")
    if any(a < b for a, b in zip(scores, scores[1:])):
        raise ValueError("Scores are not descending")

    relevant_ranks = [i for i, path in enumerate(files, 1) if path in gold]
    unranked = [path for path in labels if path not in files]
    if "unranked_correct_files" in result and set(result["unranked_correct_files"]) != set(unranked):
        raise ValueError("Saved unranked labels do not match the ranking")
    ap = sum(found / rank for found, rank in enumerate(relevant_ranks, 1)) / len(gold)
    k = 1 if len(gold) == 1 else 2
    hits = len(gold.intersection(files[:k]))
    precision, recall = hits / k, hits / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return {
        "example_index": result.get("example_index"),
        "issue_url": result.get("issue_url"),
        "repository": result.get("repository"),
        "base_sha": result.get("base_sha"),
        "correct_files": labels,
        "correct_file_count": len(gold),
        "correct_file_ranks": relevant_ranks,
        "unranked_correct_files": unranked,
        "unranked_correct_file_count": len(unranked),
        "k": k, "recall_at_k": recall, "precision_at_k": precision,
        "f1_at_k": f1, "average_precision": ap,
        "reciprocal_rank": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
        "hit_at_1": int(bool(relevant_ranks) and relevant_ranks[0] == 1),
        "hit_at_5": int(bool(relevant_ranks) and relevant_ranks[0] <= 5),
        "hit_at_10": int(bool(relevant_ranks) and relevant_ranks[0] <= 10),
    }


def aggregate(rows):
    if not rows:
        raise ValueError("No successful examples to evaluate")
    single = [row for row in rows if row["correct_file_count"] == 1]
    multiple = [row for row in rows if row["correct_file_count"] > 1]
    def average(group, key):
        return mean(row[key] for row in group) if group else None
    return {
        "evaluated_examples": len(rows),
        "single_file_examples": len(single),
        "multiple_file_examples": len(multiple),
        "examples_with_unranked_correct_files": sum(bool(row["unranked_correct_file_count"]) for row in rows),
        "unranked_correct_files_total": sum(row["unranked_correct_file_count"] for row in rows),
        "Recall_at_1_single_file": average(single, "recall_at_k"),
        "Recall_at_2_multiple_files": average(multiple, "recall_at_k"),
        "Precision_at_2_multiple_files": average(multiple, "precision_at_k"),
        "MAP_full_ranking": average(rows, "average_precision"),
        "F1_at_k_macro": average(rows, "f1_at_k"),
        "MRR": average(rows, "reciprocal_rank"),
        "Hit_at_1_all_examples": average(rows, "hit_at_1"),
        "Hit_at_5_all_examples": average(rows, "hit_at_5"),
        "Hit_at_10_all_examples": average(rows, "hit_at_10"),
    }


def evaluate(summary_path, allow_partial=False):
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    listed = summary["result_files"]
    if summary.get("configuration") != "py" or summary.get("split") != "test":
        raise ValueError("Expected the Python test split")
    if len(listed) != summary["successful"]:
        raise ValueError("Listed files do not match the summary's successful count")
    if not allow_partial and (
            summary["total_examples"] != 50 or summary["successful"] != 50
            or summary["failed"] != 0 or summary.get("remaining", 0) != 0
            or summary.get("processed") != 50):
        raise ValueError("Full evaluation requires 50 processed, 50 successful, and 0 failures. "
                         "Rerun run_all_tfidf.py; use --allow-partial only for a partial report.")
    rows, seen = [], set()
    for name in listed:
        # Accept Windows paths when the evaluator is used on another OS.
        path = (summary_path.parent / name.replace("\\", "/")).resolve()
        if not path.is_relative_to(summary_path.parent):
            raise ValueError("Result file points outside this run folder")
        if path in seen:
            raise ValueError("Duplicate result file in summary")
        seen.add(path)
        try:
            result = json.loads(path.read_text(encoding="utf-8-sig"))
            row = score_example(result)
        except Exception as error:
            raise ValueError(f"Cannot evaluate {path.name}: {error}") from error
        row["result_file"] = name
        rows.append(row)

    indices = [row["example_index"] for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate example indices")
    if not allow_partial and set(indices) != set(range(50)):
        raise ValueError("Full evaluation requires each example index from 0 through 49 exactly once")
    complete = (summary["total_examples"] == 50 and len(rows) == 50
                and summary["failed"] == 0 and summary.get("remaining", 0) == 0
                and set(indices) == set(range(50)))

    output = {
        "baseline": summary["baseline"],
        "configuration": summary["configuration"], "split": summary["split"],
        "scope": "All 50 Python test examples" if complete else "Successful examples only; partial run",
        "total_examples_in_run": summary["total_examples"],
        "excluded_failed_examples": summary["failed"],
        "remaining_examples": summary.get("remaining", 0),
        "coverage": len(rows) / summary["total_examples"],
        **aggregate(rows),
        "definitions": {
            "MAP_full_ranking": "Mean per-issue AP; divide by ALL gold files, including unretrieved ones; preserve saved tie order",
            "unranked_correct_files": "Correct files outside the candidate snapshot count as missed predictions, not execution failures",
            "F1_at_k_macro": "Mean per-issue F1 of top k; k=1 for one gold file, otherwise k=2; not verified equivalent to paper F1",
            "MRR": "Mean reciprocal rank of first correct file",
            "Hit_at_k": "Fraction of evaluated issues with at least one correct file in top k",
            "comparison": "Python test split only. The paper also evaluates Java and Kotlin; exact package/metric equivalence is not verified.",
        },
    }
    filename = "metrics_all50.json" if complete else "metrics_successes.json"
    for name, value in ((filename, output), ("per_example_metrics.json", rows)):
        (summary_path.parent / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", type=Path, help="Path to the original summary.json")
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly permit fewer than 50 examples")
    args = parser.parse_args()
    if args.summary is None:
        root = Path(__file__).resolve().parents[1] / "outputs" / "tfidf_python50"
        pointer = root / "latest_run.json"
        if pointer.exists():
            run_name = json.loads(pointer.read_text(encoding="utf-8"))["run_directory"]
            if not isinstance(run_name, str) or Path(run_name).name != run_name or run_name in (".", ".."):
                parser.error("Invalid latest-run pointer")
            args.summary = root / run_name / "summary.json"
        else:
            candidates = sorted(root.glob("*/summary.json"))
            if len(candidates) != 1:
                parser.error("Pass the summary.json path explicitly. Found:\n"
                             + ("\n".join(str(p) for p in candidates) or "No summaries"))
            args.summary = candidates[0]
    try:
        output = evaluate(args.summary, allow_partial=args.allow_partial)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print("\nTF-IDF + NLTK:", output["scope"])
    print(f"Evaluated: {output['evaluated_examples']}/{output['total_examples_in_run']}")
    print(f"Single-file: {output['single_file_examples']}; multiple-file: {output['multiple_file_examples']}")
    for key in ("Recall_at_1_single_file", "Recall_at_2_multiple_files", "Precision_at_2_multiple_files",
                "MAP_full_ranking", "F1_at_k_macro", "MRR", "Hit_at_1_all_examples",
                "Hit_at_5_all_examples", "Hit_at_10_all_examples"):
        value = output[key]
        print(f"{key}: {'N/A (no applicable examples)' if value is None else f'{value:.6f}'}")
    filename = "metrics_all50.json" if output["coverage"] == 1 else "metrics_successes.json"
    print("\nSaved:", args.summary.resolve().parent / filename)
    print("F1_at_k_macro has an explicit cutoff; it is not verified equivalent to the paper's F1.")


if __name__ == "__main__":
    main()
