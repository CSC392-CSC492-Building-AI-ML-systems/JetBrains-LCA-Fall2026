"""Score agentic runs or compare with the existing long_context rag/direct arms."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from bug_localization.agentic.run import validate_saved
from bug_localization.long_context.compare import load_run as load_context_run, metrics_for, score_results
from bug_localization.rag.retrieval import digest, project_example
from bug_localization.rag.run_ranking import contained_path, read_checked, resolve_summary
from bug_localization.rag.run_retrieval import load_dataset_input, save_json


def load_run(path, dataset):
    path = resolve_summary(path)
    manifest = json.loads((path.parent / "run_metadata.json").read_bytes())
    if manifest["mode"] != "agentic":
        return load_context_run(path, dataset)
    summary = json.loads(path.read_bytes())
    indices = manifest["example_indices"]
    if (digest(manifest) != summary["run_signature"] or manifest["dry_run"] or summary["dry_run"]
            or manifest["configuration"] != "py" or manifest["dataset_total_examples"] != len(dataset)
            or manifest["dataset_fingerprint"] != getattr(dataset, "_fingerprint", None)
            or summary["example_indices"] != indices or len(set(indices)) != len(indices)
            or summary["total_examples"] != len(indices)
            or summary["successful"] != len(summary["result_files"])
            or summary["failed"] != len(summary["failures"])
            or summary["processed"] != summary["successful"] + summary["failed"]
            or summary["total_examples"] != summary["processed"] + summary["remaining"]):
        raise ValueError("Invalid summary, dataset mismatch or dry run")
    results = {}
    for name in summary["result_files"]:
        result = read_checked(contained_path(path.parent, name))
        index = result["example_index"]
        validate_saved(result, manifest, summary["run_signature"])
        if index in results or project_example(dataset[index]) != result["example_input"]:
            raise ValueError("Duplicate result or changed input")
        results[index] = result
    failures = [r["example_index"] for r in summary["failures"]]
    if (len(set(failures)) != len(failures) or set(failures) & results.keys()
            or not set(failures).issubset(indices)
            or dict(Counter(r["status"] for r in results.values())) != summary["status_counts"]):
        raise ValueError("Inconsistent outcomes")
    return {"path": str(path), "manifest": manifest, "summary": summary, "results": results}


def resources(results):
    usages = []
    missing = 0
    for result in results:
        responses = ([t["model_response"] for t in result["trace"]] if "trace" in result
                     else ([result["model_response"]] if result["model_response"] is not None else []))
        for response in responses:
            usage = (response or {}).get("usage", {})
            if "prompt_tokens" not in usage or "completion_tokens" not in usage:
                missing += 1
            usages.append(usage)
    latencies = [r["request_latency_seconds"] for r in results]
    return {"cumulative_prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages),
            "cumulative_completion_tokens": sum(u.get("completion_tokens", 0) for u in usages),
            "cumulative_total_tokens": sum(u.get("total_tokens", u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)) for u in usages),
            "requests_missing_usage": missing,
            "model_requests": len(usages), "tool_calls": sum(r.get("tool_calls", 0) for r in results),
            "request_latency_seconds_total": sum(latencies),
            "request_latency_seconds_mean_per_issue": mean(latencies) if latencies else None,
            "elapsed_seconds_total": sum(r["elapsed_seconds"] for r in results),
            "fallback_count": sum(r["bm25_fallback"] for r in results),
            "fallback_rate": mean(r["bm25_fallback"] for r in results) if results else None}


def compare_runs(paths, dataset, output_dir, allow_partial=False):
    runs = [load_run(p, dataset) for p in paths]
    if not runs:
        raise ValueError("Supply at least one run")
    if len({r["path"] for r in runs}) != len(runs):
        raise ValueError("The same run was supplied twice")
    first = runs[0]["manifest"]
    controls = ("configuration", "split", "dataset", "dataset_fingerprint", "dataset_total_examples",
                "model", "token_counter", "candidate_policy", "retrieval", "repo_token_budget",
                "source_hashes", "packages", "python", "example_indices", "input_sha256")
    for run in runs:
        if any(run["manifest"][key] != first[key] for key in controls):
            raise ValueError("Comparison requires identical model, inputs, budgets, candidates and retrieval configuration")
        if not allow_partial and (first["split"] != "test" or first["example_indices"] != list(range(50))
                                  or set(run["results"]) != set(range(50))):
            raise ValueError("Expected all 50 py/test results; use --allow-partial for pilots")
    common = set.intersection(*(set(r["results"]) for r in runs))
    if not common:
        raise ValueError("No common completed examples")
    for index in common:
        def identity(run):
            r = run["results"][index]
            return (r["resolved_base_sha"], r["snapshot_sha256"], r["bm25_ranking"],
                    r["packing"]["allowed_candidate_paths"], r["tail_policy"])
        if any(identity(run) != identity(runs[0]) for run in runs[1:]):
            raise ValueError("Candidate snapshot or BM25 completion differs")
    model_common = {i for i in common if all(r["results"][i]["model_ranking_success"] for r in runs)}
    reports, scored_rows = [], {}
    for run in runs:
        scored = score_results(run, dataset)  # Existing score_example/aggregate; labels enter only here.
        summary = run["summary"]
        reports.append({"mode": run["manifest"]["mode"], "summary_path": run["path"],
                        "status_counts": summary["status_counts"], "selected": summary["total_examples"],
                        "completed": len(run["results"]), "failed": summary["failed"], "remaining": summary["remaining"],
                        "metrics": metrics_for(scored, common),
                        "model_success_metrics": metrics_for(scored, model_common),
                        "resources_common": resources([run["results"][i] for i in sorted(common)]),
                        "resources_run": resources(list(run["results"].values()))})
        scored_rows[run["path"]] = list(scored.values())
    output = {"configuration": "py", "split": first["split"], "model": first["model"],
              "repo_token_budget": first["repo_token_budget"], "common_indices": sorted(common),
              "common_model_success_indices": sorted(model_common), "runs": reports,
              "scope": "Pipeline scores include full BM25 fallbacks. Excluded gold files remain misses.",
              "usage_note": "Cumulative API-reported usage across turns; missing usage is counted, never inferred as free. Failed retry usage may be unavailable."}
    columns = ["Mode", "MAP", "MRR", "Hit@1", "Hit@5", "Hit@10", "Tokens", "Request seconds", "Fallback rate", "Scored/selected", "Failed", "Remaining"]
    lines = ["# Bug localization comparison", "", output["scope"], "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for report in reports:
        m, r = report["metrics"], report["resources_common"]
        values = [report["mode"], *[f"{m[k]:.6f}" for k in ("MAP_full_ranking", "MRR", "Hit_at_1_all_examples", "Hit_at_5_all_examples", "Hit_at_10_all_examples")],
                  str(r["cumulative_total_tokens"]), f"{r['request_latency_seconds_total']:.3f}",
                  f"{r['fallback_rate']:.3f}", f"{len(common)}/{report['selected']}", str(report["failed"]), str(report["remaining"])]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", output["usage_note"], "", "JSON includes prompt/completion totals, per-issue mean latency, wall time, run-wide resources and common model-success sensitivity metrics."]
    folder = Path(output_dir)
    save_json(folder / "comparison.json", output)
    save_json(folder / "per_example_metrics.json", scored_rows)
    (folder / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "comparison")
    args = parser.parse_args(argv)
    summary = json.loads(resolve_summary(args.runs[0]).read_bytes())
    dataset = load_dataset_input(summary["split"], args.dataset_cache_dir, args.dataset_file)
    compare_runs(args.runs, dataset, args.output_dir, args.allow_partial)
    print((args.output_dir / "comparison.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
