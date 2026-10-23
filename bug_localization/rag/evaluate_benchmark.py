"""Step-5 reporting over saved, validated predictions; no inference."""

import csv
import io
import json
import tempfile
from pathlib import Path
from statistics import mean

from bug_localization.baseline_eval.evaluate_tfidf import aggregate, score_example
from bug_localization.rag import evaluate_ranking
from bug_localization.rag.run_benchmark import LABEL_POLICY
from bug_localization.rag.run_ranking import contained_path, resolve_summary
from bug_localization.rag.run_retrieval import save_json


def write_comparison(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    with tempfile.TemporaryDirectory(prefix=".writing-", dir=path.parent) as work:
        temporary = Path(work) / path.name
        temporary.write_text(stream.getvalue(), encoding="utf-8")
        temporary.replace(path)


def evaluate(summary_path, dataset, tfidf_summary=None):
    path = resolve_summary(summary_path)
    folder = path.parent
    manifest = json.loads((folder / "run_metadata.json").read_bytes())
    if manifest.get("benchmark_version") != 1:
        raise ValueError("Expected a step-5 benchmark run")
    output = evaluate_ranking.evaluate_run(path, dataset)
    details = json.loads((folder / "evaluation" / "per_example_metrics.json").read_bytes())
    indices = [d["example_index"] for d in details]
    complete = len(indices) == 50 and set(indices) == set(range(50))
    scope = "All 50 Python test examples" if complete else "Partial run; completed predictions only"
    output.update(scope=scope, complete_50_example_evaluation=complete, evaluated_indices=indices,
                  missing_prediction_indices=sorted(set(manifest["example_indices"]) - set(indices)),
                  dataset_total_examples=50, selected_examples=len(manifest["example_indices"]),
                  model_response_examples=sum(output["status_counts"].get(s, 0) for s in
                      ("success", "repaired", "fallback_request", "fallback_validation")))
    for phase in ("before_packing", "after_packing"):
        eligible = [d[phase] for d in details if d[phase]["candidate_file_recall"] is not None]
        coverage = {"denominator": len(eligible), "excluded_zero_eligible_gold": len(details) - len(eligible)}
        for key in ("at_least_one_eligible_retrieved", "all_eligible_retrieved"):
            coverage[key] = mean(d[key] for d in eligible) if eligible else None
        # Also expose coverage against ALL raw gold files, including unreachable labels.
        raw = []
        for d in details:
            metric = d[phase]
            gold_count = metric["eligible_ground_truth_count"] + len(metric["ground_truth_outside_candidates"])
            hits = metric["retrieved_eligible_count"]
            raw.append((hits / gold_count, int(hits > 0), int(hits == gold_count)))
        coverage["raw_gold_denominator"] = len(raw)
        for pos, key in enumerate(("raw_gold_recall_macro", "at_least_one_raw_gold", "all_raw_gold")):
            coverage[key] = mean(r[pos] for r in raw) if raw else None
        output["retrieval_coverage_" + phase] = coverage
    output["aggregate_denominators"] = {
        "ranking": "Each approach records evaluated_examples, single_file_examples and multiple_file_examples",
        "coverage": "Each retrieval phase records eligible-gold and all-raw-gold denominators",
        "response_counts": len(details), "latency": output["model_response_examples"],
        "usage": output["usage_reporting_examples"], "selected": len(manifest["example_indices"])}

    comparison_indices = []
    comparison_metrics = {}
    baseline_status = "unavailable: no TF-IDF summary supplied"
    if tfidf_summary is not None:
        try:
            baseline_path = resolve_summary(tfidf_summary)
            baseline_summary = json.loads(baseline_path.read_bytes())
            baseline_manifest = json.loads((baseline_path.parent / "run_metadata.json").read_bytes())
            if (baseline_manifest.get("dataset") != manifest["dataset"]
                    or baseline_manifest.get("label_policy") != LABEL_POLICY
                    or baseline_summary.get("label_policy") != LABEL_POLICY
                    or not manifest["dataset_fingerprint"]):
                raise ValueError("Dataset or label policy is incompatible/unverifiable")
            baseline = evaluate_ranking._tfidf_results(baseline_path, manifest)
            scored = {name: [] for name in ("TF-IDF", "BM25", "BM25 + LLM RAG")}
            for index in indices:
                if index not in baseline:
                    continue
                bm25 = json.loads((folder / "evaluation" / "bm25" / f"{index:03d}.json").read_bytes())
                rag = json.loads((folder / "evaluation" / "rag" / f"{index:03d}.json").read_bytes())
                tfidf = baseline[index]
                if (any(tfidf[k] != bm25[k] for k in ("issue_url", "repository", "base_sha", "candidate_count"))
                        or set(tfidf["correct_files"]) != set(bm25["correct_files"])
                        or {r["file"] for r in tfidf["ranking"]} != {r["file"] for r in bm25["ranking"]}):
                    raise ValueError("TF-IDF identity, labels or candidate universe mismatch")
                for name, result in (("TF-IDF", tfidf), ("BM25", bm25), ("BM25 + LLM RAG", rag)):
                    scored[name].append(score_example(result))
                comparison_indices.append(index)
            comparison_metrics = {name: aggregate(rows) for name, rows in scored.items() if rows}
            baseline_status = "compatible" if comparison_indices else "unavailable: no common predictions"
        except FileNotFoundError:
            baseline_status = "unavailable: TF-IDF artifacts not found"
            comparison_indices, comparison_metrics = [], {}
        except (OSError, ValueError, KeyError, TypeError):
            baseline_status = "incompatible: dataset, policy, identity, candidates or artifact validation failed"
            comparison_indices, comparison_metrics = [], {}
    output["baseline_status"] = baseline_status
    output["comparison_indices"] = comparison_indices
    output["comparison_metrics"] = comparison_metrics
    output["comparison"] = ("All 50 matched examples" if len(comparison_indices) == 50 else
                            "Common-example comparison; partial" if comparison_indices else baseline_status)
    csv_rows = []
    for name in ("TF-IDF", "BM25", "BM25 + LLM RAG"):
        values = comparison_metrics.get(name) if comparison_indices else output["file_ranking_metrics"].get(name)
        csv_rows.append({"approach": name, "status": baseline_status if name == "TF-IDF" else "available",
                         "scope": output["comparison"] if comparison_indices else scope,
                         "example_indices": json.dumps(comparison_indices if comparison_indices else indices)
                         if values else "[]", "selected_examples": output["selected_examples"],
                         "evaluated_examples": 0, **(values or {})})
    write_comparison(folder / "comparison.csv", csv_rows)
    save_json(folder / "metrics.json", output)
    return output
