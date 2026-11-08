"""Score saved predictions with the existing evaluator and compare matched examples."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from bug_localization.baseline_eval.evaluate_tfidf import aggregate, score_example
from bug_localization.baseline_eval.run_all_tfidf import get_labels
from bug_localization.long_context.experiment import COMPLETE_STATUSES, MODEL_STATUSES, TAIL_POLICY, validate_saved
from bug_localization.rag.retrieval import digest, project_example
from bug_localization.rag.run_ranking import contained_path, read_checked, resolve_summary
from bug_localization.rag.run_retrieval import load_dataset_input, save_json


METRICS = {"MAP": "MAP_full_ranking", "MRR": "MRR", "Hit@1": "Hit_at_1_all_examples",
           "Hit@5": "Hit_at_5_all_examples", "Hit@10": "Hit_at_10_all_examples"}


def load_run(summary_path, dataset):
    path = resolve_summary(summary_path)
    summary = json.loads(path.read_bytes())
    manifest = json.loads((path.parent / "run_metadata.json").read_bytes())
    indices = manifest["example_indices"]
    if (digest(manifest) != summary["run_signature"] or manifest["dry_run"]
            or manifest["configuration"] != "py" or summary["dry_run"]
            or manifest["dataset_total_examples"] != len(dataset)
            or manifest["dataset_fingerprint"] != getattr(dataset, "_fingerprint", None)
            or summary["example_indices"] != indices or len(set(indices)) != len(indices)
            or len(indices) != summary["total_examples"]
            or summary["successful"] != len(summary["result_files"])
            or summary["failed"] != len(summary["failures"])
            or summary["processed"] != summary["successful"] + summary["failed"]
            or summary["total_examples"] != summary["processed"] + summary["remaining"]):
        raise ValueError("Invalid run summary, dataset mismatch or unscorable dry run")
    results = {}
    for name in summary["result_files"]:
        result = read_checked(contained_path(path.parent, name))
        index = result["example_index"]
        if index in results or result["status"] not in COMPLETE_STATUSES:
            raise ValueError("Duplicate or incomplete prediction")
        validate_saved(result, manifest, summary["run_signature"])
        if project_example(dataset[index]) != result["example_input"]:
            raise ValueError("Dataset prediction inputs differ")
        results[index] = result
    failed = [f["example_index"] for f in summary["failures"]]
    if (len(set(failed)) != len(failed) or set(failed) & results.keys()
            or not set(failed).issubset(indices)
            or dict(Counter(r["status"] for r in results.values())) != summary["status_counts"]):
        raise ValueError("Inconsistent failures or status counts")
    return {"path": str(path), "summary": summary, "manifest": manifest, "results": results}


def score_results(run, dataset):
    """Labels enter only this scoring adapter. Unreachable labels remain gold."""
    scored = {}
    for index, result in run["results"].items():
        example = result["example_input"]
        labels = get_labels(dataset[index]["changed_files"])
        row = score_example({
            "status": "success", "example_index": index, "issue_url": example["issue_url"],
            "repository": f"{example['repo_owner']}__{example['repo_name']}",
            "base_sha": result["resolved_base_sha"], "candidate_count": result["candidate_count"],
            "correct_files": labels, "ranking": result["ranking"],
        })
        source_paths = set(result["packing"]["included_files"])
        row.update(prediction_status=result["status"], bm25_fallback=result["bm25_fallback"],
                   gold_files_in_source=sum(p in source_paths for p in labels),
                   gold_source_recall=sum(p in source_paths for p in labels) / len(labels))
        scored[index] = row
    return scored


def metrics_for(scored, indices):
    return aggregate([scored[i] for i in sorted(indices)]) if indices else None


def table(reports, key, title):
    lines = [title, "", "| Mode | Source budget | MAP | MRR | Hit@1 | Hit@5 | Hit@10 | Scored / selected | Scored / dataset | Model OK | Fallbacks | Pipeline failures | Request failures | Invalid responses | Context failures | Remaining |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        metrics = report[key]
        values = [f"{metrics[name]:.6f}" if metrics else "N/A" for name in METRICS.values()]
        n = metrics["evaluated_examples"] if metrics else 0
        counts = report["status_counts"]
        lines.append("| " + " | ".join([
            report["mode"], str(report["repo_token_budget"]), *values,
            f"{n}/{report['selected_examples']}", f"{n}/{report['dataset_total_examples']}",
            str(report["successful_model_responses"]), str(report["fallbacks"]),
            str(report["pipeline_failures"]), str(counts.get("fallback_request", 0)),
            str(counts.get("fallback_validation", 0)), str(counts.get("fallback_context", 0)),
            str(report["remaining_examples"]),
        ]) + " |")
    return "\n".join(lines)


def compare_runs(summary_paths, dataset, output_dir):
    if not summary_paths:
        raise ValueError("Supply at least one run")
    runs = [load_run(p, dataset) for p in summary_paths]
    if len({r["summary"]["run_signature"] for r in runs}) != len(runs):
        raise ValueError("The same run was supplied twice")
    first = runs[0]["manifest"]
    # Mode, source budget, and selected subset are the only permitted differences.
    controls = ("experiment_version", "configuration", "split", "dataset", "dataset_fingerprint",
                "dataset_total_examples", "model", "token_counter", "candidate_policy", "retrieval",
                "source_hashes", "packages", "python")
    for run in runs[1:]:
        if any(run["manifest"][key] != first[key] for key in controls):
            raise ValueError("Comparison requires identical model, data, tokenizer, implementation and retrieval settings")
        shared = set(first["example_indices"]) & set(run["manifest"]["example_indices"])
        if any(first["input_sha256"][str(i)] != run["manifest"]["input_sha256"][str(i)] for i in shared):
            raise ValueError("Prediction inputs differ across runs")
    common = set.intersection(*(set(run["results"]) for run in runs))
    model_common = {i for i in common if all(run["results"][i]["status"] in MODEL_STATUSES for run in runs)}
    seen = {}
    for run in runs:
        for index, result in run["results"].items():
            identity = (result["resolved_base_sha"], result["snapshot_sha256"],
                        result["packing"]["allowed_candidate_paths"], result["bm25_ranking"])
            if index in seen and seen[index] != identity:
                raise ValueError("Snapshot, candidate universe or BM25 tail differs across runs")
            seen[index] = identity
    reports = []
    per_example = {}
    for run in runs:
        scored = score_results(run, dataset)
        summary = run["summary"]
        counts = summary["status_counts"]
        reports.append({
            "run_signature": summary["run_signature"], "summary_path": run["path"],
            "mode": run["manifest"]["mode"], "repo_token_budget": run["manifest"]["repo_token_budget"],
            "selected_examples": summary["total_examples"], "dataset_total_examples": len(dataset),
            "evaluated_example_indices": sorted(scored), "status_counts": counts,
            "pipeline_failures": summary["failed"], "remaining_examples": summary["remaining"],
            "successful_model_responses": sum(counts.get(k, 0) for k in MODEL_STATUSES),
            "fallbacks": sum(n for k, n in counts.items() if k.startswith("fallback_")),
            "available_metrics": metrics_for(scored, set(scored)),
            "common_metrics": metrics_for(scored, common),
            "common_model_success_metrics": metrics_for(scored, model_common),
            "request_latency_seconds_total": sum(r["request_latency_seconds"] for r in run["results"].values()),
        })
        per_example[summary["run_signature"]] = list(scored.values())
    output = {
        "dataset": first["dataset"], "configuration": first["configuration"], "split": first["split"],
        "scope": "Pipeline rankings include explicit BM25 fallbacks; matched tables use a common example set",
        "tail_policy": TAIL_POLICY, "common_example_indices": sorted(common),
        "common_model_success_example_indices": sorted(model_common), "runs": reports,
        "definitions": {
            "metrics": "Unchanged baseline_eval.evaluate_tfidf.score_example and aggregate",
            "labels": "All raw changed_files, including files excluded by candidate filters, count in AP denominators",
            "coverage": "Every score states its evaluated count / selected count and / full dataset count",
            "model_success": "success or repaired; partial valid responses append BM25 tail; raw response and repairs saved",
            "failure_counts": "Run-wide counts, also shown beside matched subset metrics; remaining examples are not failures",
            "incomplete": "Available-only scores are descriptive; use the common-example table for head-to-head comparisons",
        },
    }
    markdown = "\n\n".join([
        "Controlled source-context comparison",
        "Pipeline scores include BM25 fallbacks. Counts of model outcomes and failures are run-wide in every table.",
        table(reports, "available_metrics", "Available predictions per run (coverage can differ):"),
        table(reports, "common_metrics", f"Head-to-head: common completed examples {sorted(common)}:"),
        table(reports, "common_model_success_metrics", f"Sensitivity: common model-success examples {sorted(model_common)}:"),
        "No common examples yields N/A, never zero-valued benchmark scores. Ground-truth files outside candidates remain misses.",
        "Shared tail: " + TAIL_POLICY + ". issue_only sends zero source tokens to the model; its final ranking can include BM25-derived paths.",
    ]) + "\n"
    folder = Path(output_dir)
    save_json(folder / "comparison.json", output)
    save_json(folder / "per_example_metrics.json", per_example)
    (folder / "comparison.md").write_text(markdown, encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="summary.json, run folder, or latest-run parent")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "comparison")
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--dataset-cache-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = json.loads(resolve_summary(args.runs[0]).read_bytes())
        dataset = load_dataset_input(summary["split"], args.dataset_cache_dir, args.dataset_file)
        compare_runs(args.runs, dataset, args.output_dir)
        print((args.output_dir / "comparison.md").read_text(encoding="utf-8"))
        print(f"Outputs: {args.output_dir.resolve()}")
        return 0
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
