"""Verify completed real API artifacts without inference or rescoring."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked(path):
    value = read(path)
    checksum = value.pop("artifact_sha256", None)
    if digest(value) != checksum:
        raise ValueError(f"Checksum mismatch: {path.name}")
    return value


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify(run):
    manifest, report = read(run / "manifest.json"), read(run / "comparison.json")
    scores = read(run / "per_example_scores.json")
    indices = report["common_indices"]
    if not report["complete"] or indices != list(range(report["selected_examples"])):
        raise ValueError("Publication requires complete matched coverage")
    original_id = digest(manifest)
    arms = manifest["arms"]
    contexts, keys = {}, set()
    for arm in arms:
        if len(list((run / "arms" / arm).glob("*.json"))) != len(indices):
            raise ValueError(f"Unexpected context coverage: {arm}")
        for index in indices:
            context = checked(run / "arms" / arm / f"{index}.json")
            if context["arm"] != arm or context["index"] != index:
                raise ValueError("Context identity mismatch")
            if context["actual_context_tokens"] > context["requested_context_tokens"] or context["estimated_input_tokens"] > context["input_budget"]:
                raise ValueError("Saved context exceeds its declared budget")
            contexts[(arm, index)] = context
            keys.add(context["request_id"])
    requests = {}
    for path in (run / "requests").glob("*.json"):
        record = checked(path)
        if digest(record["request"]) != path.stem or record["request"]["experiment"] != original_id:
            raise ValueError("Request identity differs from manifest")
        if record["status"] != "success" or record["returned_model"] != manifest["settings"]["model"]:
            raise ValueError("Incomplete prediction or model mismatch")
        if digest(record["response"]) != record["response_sha256"]:
            raise ValueError("Response checksum mismatch")
        requests[path.stem] = record
    if keys != set(requests):
        raise ValueError("Unreferenced/missing physical request artifacts")
    by_arm = {}
    for arm in arms:
        values = {r["index"]: r for r in scores[arm]}
        if set(values) != set(indices) or len(scores[arm]) != len(indices):
            raise ValueError("Duplicate or missing per-task scores")
        by_arm[arm] = values
        for index in indices:
            context = contexts[(arm, index)]
            record = requests[context["request_id"]]
            if record["request"]["index"] != index or record["request"]["messages"] != [{"role": "user", "content": context["prompt"]}]:
                raise ValueError("Request does not match context")
            if values[index]["request_id"] != context["request_id"]:
                raise ValueError("Scores refer to another request")
        for metric in ("ChrF", "API_recall"):
            if not math.isclose(mean(values[i][metric] for i in indices), report["arms"][arm]["metrics"][metric], abs_tol=1e-12):
                raise ValueError("Published mean differs from per-task scores")
    for name, result in report["contrasts"].items():
        arm, baseline = name.split("_minus_")
        for metric in ("ChrF", "API_recall"):
            expected = mean(by_arm[arm][i][metric] - by_arm[baseline][i][metric] for i in indices)
            if not math.isclose(expected, result["metrics"][metric]["difference"], abs_tol=1e-12):
                raise ValueError("Paired difference differs from archived scores")
    reused = 0
    for i in indices:
        reused += contexts[("direct_8k", i)]["request_id"] == contexts[("full_list", i)]["request_id"]
        if contexts[("bm25_20", i)]["request_id"] == contexts[("bm25_20_repeat", i)]["request_id"]:
            raise ValueError("Independent repeat improperly reused baseline response")
    sources = {}
    for relative, expected in manifest["source_hashes"].items():
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("Source path escapes repository")
        sources[relative] = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    if not all(sources.values()):
        raise ValueError("Generation/scoring sources differ from saved manifest")
    usage = [r["usage"] for r in requests.values()]
    prompt = sum(u["prompt_tokens"] for u in usage)
    completion = sum(u["completion_tokens"] for u in usage)
    cost = sum(r["cost_micro_usd"] for r in requests.values()) / 1e6
    latency = sum(r["latency_seconds"] for r in requests.values())
    for field, actual in (("saved", len(requests)), ("successful", len(requests)), ("prompt_tokens", prompt), ("completion_tokens", completion)):
        if report["physical_requests"][field] != actual:
            raise ValueError("Physical request totals disagree with report")
    if not math.isclose(report["physical_requests"]["known_cost_usd"], cost, abs_tol=1e-12):
        raise ValueError("List-price cost total mismatch")
    groups = Counter(row["repo"] for row in scores[arms[0]])
    if len(groups) != report["repository_count"]:
        raise ValueError("Repository denominator mismatch")
    shortfalls = [contexts[("random_8k", i)]["token_shortfall"] for i in indices]
    created = [r["raw_response"].get("created") for r in requests.values() if r["raw_response"].get("created") is not None]
    diagnostics = {
        "unique_requests": len(requests), "logical_predictions": len(contexts),
        "shared_direct_full_responses": reused,
        "statuses": dict(Counter(r["status"] for r in requests.values())),
        "finish_reasons_physical": dict(Counter(r["finish_reason"] for r in requests.values())),
        "provider_prompt_tokens": prompt, "provider_completion_tokens": completion,
        "provider_cached_prompt_tokens": sum((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0 for u in usage),
        "uncached_list_price_accounting_usd": cost,
        "cost_note": "Runner charges cached input at uncached prices; this is not a verified invoice.",
        "summed_request_latency_seconds": latency,
        "median_request_latency_seconds": median(r["latency_seconds"] for r in requests.values()),
        "wall_time_note": "Summed latency excludes preparation, scoring, pauses and file IO; wall time was not measured.",
        "provider_first_created_utc": datetime.fromtimestamp(min(created), timezone.utc).isoformat() if created else None,
        "provider_last_created_utc": datetime.fromtimestamp(max(created), timezone.utc).isoformat() if created else None,
        "repository_sizes": dict(groups), "random_match_max_shortfall_tokens": max(shortfalls),
        "random_match_mean_shortfall_tokens": mean(shortfalls),
        "generation_scoring_source_hashes_match": sources,
    }
    verification = {
        "status": "passed", "original_run_id": original_id[:20], "archived_directory": run.name,
        "logical_predictions_checked": len(contexts), "physical_responses_checked": len(requests),
        "artifact_and_response_checksums": "passed", "request_context_mapping": "passed",
        "aggregate_means_and_paired_differences": "passed", "saved_budget_bounds": "passed",
        "independent_repeat": "passed", "source_hashes": "passed",
        "scope": "Saved artifacts verified; official scores and bootstrap intervals were not recomputed. Provider records are not cryptographically authenticated.",
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write(run / "diagnostics.json", diagnostics)
    write(run / "verification.json", verification)
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).parent / "results/experiment")
    args = parser.parse_args()
    diagnostics = verify(args.run_dir)
    print(json.dumps({k: v for k, v in diagnostics.items() if k not in ("repository_sizes", "generation_scoring_source_hashes_match")}, indent=2))


if __name__ == "__main__":
    main()
