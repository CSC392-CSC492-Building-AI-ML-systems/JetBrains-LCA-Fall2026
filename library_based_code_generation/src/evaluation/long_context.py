"""Audit-gated API-list experiment. Default: prepare and estimate; no inference."""

import argparse
import csv
from collections import Counter
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
from pathlib import Path
import random
from statistics import mean
import time

from ..context.long_context import pack_context, HINT
from .spending import SpendingLedger
from .rag_analysis import extract_code, paired_stats

DATASET = "JetBrains-Research/lca-library-based-code-generation"
REVISION = "dc460b7e403f63e58ec7128b80eaf2c9c95344f8"
ARMS = ("plain", "bm25_20", "direct_8k", "full_list", "bm25_8k", "random_8k", "bm25_20_repeat")
PRIMARY = (("full_list", "direct_8k"), ("bm25_8k", "direct_8k"), ("bm25_8k", "random_8k"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sealed(value):
    return {**value, "artifact_sha256": digest(value)}


def read_record(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.pop("artifact_sha256", None)
    if expected != digest(value):
        raise ValueError(f"Artifact checksum mismatch: {path.name}")
    return value


def paired_interval(differences, draws=10000):
    return paired_stats(differences, list(range(len(differences))), draws)["ci95"]


def official_order(apis, instruction):
    """Preserve official candidate duplicates, tokenizer and reverse-argsort ties."""
    import numpy as np
    from rank_bm25 import BM25Okapi
    from ..models.utils import split_identifier
    corpus = [split_identifier(a) for a in apis]
    if not corpus or not any(corpus):
        return list(reversed(apis))
    query = "".join(c for c in instruction if c.isalnum() or c == " ").lower().split(" ")
    return [apis[i] for i in reversed(np.argsort(BM25Okapi(corpus).get_scores(query))) ]


def charge_micro(input_tokens, output_tokens, settings):
    return math.ceil(input_tokens * settings["input_usd_per_million"] + output_tokens * settings["output_usd_per_million"])


def request_for(index, task, packed, settings, experiment, repeat=False):
    parameters = {"model": settings["model"], "temperature": 0, "top_p": 1,
                  "max_tokens": settings["max_output_tokens"], "frequency_penalty": 0, "presence_penalty": 0}
    request = {"experiment": experiment, "index": index, "task_sha256": digest(task),
               "replicate": "independent_repeat" if repeat else "main",
               "messages": [{"role": "user", "content": packed["prompt"]}], "parameters": parameters}
    return digest(request), request


def prepare(rows, settings, count, experiment, ranker=official_order):
    contexts, requests, audit = {}, {}, []
    for index, row in enumerate(rows):
        print(f"Preparing task {index + 1}/{len(rows)}", flush=True)
        # Explicit allowlist: no labels or reference source reach selection.
        task = {key: row[key] for key in ("instruction", "project_defined_elements")}
        apis = list(task["project_defined_elements"])
        ranked = ranker(apis, task["instruction"])
        if Counter(apis) != Counter(ranked):
            raise ValueError("Ranker changed the candidate inventory")
        common = dict(count=count, context_window=settings["context_window"],
                      max_output_tokens=settings["max_output_tokens"], safety_tokens=settings["safety_tokens"],
                      deduplicate=False)

        def pack(order, budget, plain=False):
            return pack_context(task["instruction"], apis, "plain" if plain else "direct", budget,
                                ordered_apis=order, **common)

        plain = pack([], 0, True)
        top20 = pack(ranked[:20], settings["full_budget"])
        if top20["included_count"] != len(ranked[:20]):
            raise ValueError("Official top-20 does not fit; do not silently truncate baseline")
        direct = pack(apis, settings["small_budget"])
        full = pack(apis, settings["full_budget"])
        if full["truncated"]:
            raise ValueError("Full API list does not fit configured limits; revise protocol before inference")
        bm25 = pack(ranked, settings["small_budget"])
        shuffled = list(apis)
        random.Random(int(digest({"seed": settings["random_seed"], "task": task}), 16)).shuffle(shuffled)
        target = bm25["actual_context_tokens"]
        placebo = pack(shuffled, target)
        placebo["matched_target_tokens"] = target
        placebo["token_shortfall"] = target - placebo["actual_context_tokens"]
        placebo["matching_tolerance_tokens"] = max(8, math.ceil(target * .02))
        placebo["match_within_tolerance"] = placebo["token_shortfall"] <= placebo["matching_tolerance_tokens"]
        if not placebo["match_within_tolerance"]:
            raise ValueError("Random evidence cannot match target within frozen tolerance; revise preparation before inference")
        selected = dict(zip(ARMS, (plain, top20, direct, full, bm25, placebo, top20)))
        for arm, packed in selected.items():
            if hasattr(count, "verify"):
                count.verify(packed, HINT)
            request_id, request = request_for(index, task, packed, settings, experiment, arm.endswith("_repeat"))
            requests[request_id] = request
            contexts[(arm, index)] = {"index": index, "arm": arm, "request_id": request_id, **packed}
        gold = set(row["unique_apis"])
        audit.append({"index": index, "repo": row.get("repo_full_name"),
                      "full_fit_at_small_budget": direct["selected_apis"] == full["selected_apis"],
                      "full_list_hint_recall": len(gold & set(apis)) / len(gold) if gold else None,
                      "bm25_at20_hint_recall": len(gold & set(ranked[:20])) / len(gold) if gold else None})
    estimates = []
    for key, req in requests.items():
        prompt = req["messages"][0]["content"]
        estimated = count(prompt) + 32
        conservative = len(prompt.encode("utf-8")) + settings["safety_tokens"] + 256
        estimates.append({"request_id": key, "estimated_input_tokens": estimated,
                          "reserved_input_tokens": max(estimated, conservative),
                          "prices": {k: settings[k] for k in ("input_usd_per_million", "output_usd_per_million")},
                          "estimated_max_cost_micro_usd": charge_micro(estimated, settings["max_output_tokens"], settings),
                          "reservation_micro_usd": charge_micro(max(estimated, conservative), settings["max_output_tokens"], settings)})
    return contexts, requests, estimates, audit


def execute(run, requests, estimates, ledger, client_factory, expected_model):
    """One durable response per exact request. Repeat keys never share main keys."""
    client = None
    costs = {r["request_id"]: r for r in estimates}
    for key, request in requests.items():
        path = run / "requests" / f"{key}.json"
        if path.exists():
            existing = read_record(path)
            if existing["request"] != request:
                raise ValueError("Saved request identity mismatch")
            if existing["status"] == "success":
                if digest(existing["response"]) != existing.get("response_sha256") or existing.get("returned_model") != expected_model:
                    raise ValueError("Saved response checksum/snapshot mismatch")
                continue
            raise RuntimeError("Unfinished or failed request needs manual billing reconciliation; no automatic retry")
        if client is None:
            client = client_factory()
        ledger.reserve(key, costs[key]["reservation_micro_usd"])
        record = {"request": request, "status": "requesting"}
        save(path, sealed(record))
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(messages=request["messages"], **request["parameters"])
            record.update(raw_response=response.model_dump(), returned_model=response.model,
                          finish_reason=response.choices[0].finish_reason,
                          response=response.choices[0].message.content,
                          latency_seconds=time.perf_counter() - started)
            record["usage"] = response.usage.model_dump() if response.usage else None
            # Persist the response before settling so interrupted accounting is recoverable.
            record["status"] = "response_received"
            save(path, sealed(record))
            if response.usage:
                actual = charge_micro(response.usage.prompt_tokens, response.usage.completion_tokens, costs[key]["prices"])
                record["cost_micro_usd"] = actual
                ledger.settle(key, actual)
            else:
                record["cost_micro_usd"] = None
                record["billing_status"] = "usage_missing_reservation_retained"
            if response.model != expected_model:
                raise ValueError("Returned model differs from frozen snapshot")
            if response.choices[0].finish_reason not in ("stop", "length"):
                raise ValueError("Filtered or unsupported completion outcome")
            if getattr(response.choices[0].message, "refusal", None) or not isinstance(record["response"], str) or not record["response"].strip():
                raise ValueError("Empty or refused response")
            record["status"] = "success"
            record["response_sha256"] = digest(record["response"])
            save(path, sealed(record))
        except BaseException as error:
            record.update(status="failed", error_type=type(error).__name__)
            save(path, sealed(record))
            raise


def call_diagnostics(calls, row):
    calls, library, gold = set(calls), set(row["project_defined_elements"]), set(row["unique_apis"])
    listed = calls & library
    return {"unique_called_names": len(calls), "used_library_names": len(listed),
            "extra_library_names": sorted(listed - gold),
            "extra_library_fraction": len(listed - gold) / len(listed) if listed else None,
            "unlisted_names": sorted(calls - library),
            "unlisted_fraction": len(calls - library) / len(calls) if calls else None,
            "reference_absent_from_hints": sorted(gold - library),
            "reference_in_hints_but_unused": sorted((gold & library) - calls)}


def analyze(run, rows, contexts, settings, scorer=None, draws=10000):
    if scorer is None:
        from ..metrics.chrf import ChrF
        from ..metrics.overlap import Overlap
        from ..context.parsed_file import ParsedFile
        chrf, recall = ChrF(), Overlap()

        def scorer(code, row):
            return {"ChrF": float(chrf.score(code, row["clean_reference"], row["unique_apis"])),
                    "API_recall": float(recall.score(code, row["clean_reference"], row["unique_apis"]))}, ParsedFile(code=code).called_functions
    scores = {arm: {} for arm in ARMS}
    records, request_cache = {}, {}
    for (arm, index), context in contexts.items():
        key = context["request_id"]
        path = run / "requests" / f"{key}.json"
        if not path.exists():
            continue
        if key not in request_cache:
            request_cache[key] = read_record(path)
        record = request_cache[key]
        if digest(record["request"]) != key or record["request"]["messages"] != [{"role": "user", "content": context["prompt"]}]:
            raise ValueError("Saved request does not match prepared context")
        records[key] = record
        if record["status"] != "success":
            continue
        if digest(record["response"]) != record["response_sha256"] or record["returned_model"] != settings["model"]:
            raise ValueError("Prediction checksum or snapshot mismatch")
        row = rows[index]
        metrics, calls = scorer(extract_code(record["response"]), row)
        hint_row = {**row, "project_defined_elements": context["selected_apis"]}
        diagnostics = call_diagnostics(calls, row)
        hint_diag = call_diagnostics(calls, hint_row)
        scores[arm][index] = {"index": index, "repo": row.get("repo_full_name"), **metrics, **diagnostics,
                              "reference_absent_from_hints": hint_diag["reference_absent_from_hints"],
                              "reference_in_hints_but_unused": hint_diag["reference_in_hints_but_unused"],
                              "context_tokens": context["actual_context_tokens"], "request_id": key}
    common = sorted(set.intersection(*(set(v) for v in scores.values())))
    grouped = all(isinstance(r.get("repo_full_name"), str) and r["repo_full_name"] for r in rows)
    groups = [rows[i]["repo_full_name"] if grouped else str(i) for i in common]
    contrasts = (*PRIMARY, ("bm25_20_repeat", "bm25_20"), ("full_list", "bm25_20"), ("bm25_20", "plain"))
    report = {"selected_examples": len(rows), "common_indices": common, "complete": len(common) == len(rows),
              "full_split": settings["full_split"], "bootstrap_unit": "repository" if grouped else "example (repository identity unavailable)",
              "repository_count": len(set(groups)), "bootstrap_draws": draws, "tie_tolerance": 1e-8,
              "meaningful_api_recall_difference": .02, "arms": {}, "contrasts": {},
              "hypothesis_tests": "No p-values/rejection claims; primary intervals are descriptive, unadjusted for multiplicity.",
              "diagnostic_limitations": "Unlisted calls include builtins/local/external functions; bare-name matching is not semantic correctness."}
    for arm, values in scores.items():
        actual = [values[i] for i in common]
        diagnostics = {}
        for metric in ("extra_library_fraction", "unlisted_fraction"):
            finite = [v[metric] for v in actual if v[metric] is not None]
            diagnostics[metric] = {"mean": mean(finite) if finite else None, "denominator": len(finite)}
        report["arms"][arm] = {"completed": len(values), "remaining": len(rows) - len(values),
                               "metrics": {m: mean(v[m] for v in actual) if actual else None for m in ("ChrF", "API_recall")},
                               "call_diagnostics": diagnostics,
                               "mean_context_tokens": mean(contexts[(arm, i)]["actual_context_tokens"] for i in common) if common else None,
                               "length_limited": sum(records[contexts[(arm, i)]["request_id"]].get("finish_reason") == "length" for i in values)}
    for arm, baseline in contrasts:
        result = {"primary": (arm, baseline) in PRIMARY, "metrics": {}, "strata": {}}
        for metric in ("ChrF", "API_recall"):
            diffs = [scores[arm][i][metric] - scores[baseline][i][metric] for i in common]
            stats = paired_stats(diffs, groups, draws) if common else None
            if stats:
                stats.update(wins=sum(d > 1e-8 for d in diffs), losses=sum(d < -1e-8 for d in diffs),
                             ties=sum(abs(d) <= 1e-8 for d in diffs),
                             changes_at_least_002=sum(abs(d) >= .02 for d in diffs))
            result["metrics"][metric] = stats
        for full_fit in (True, False):
            indices = [i for i in common if (contexts[("direct_8k", i)]["selected_apis"] == contexts[("full_list", i)]["selected_apis"]) == full_fit]
            result["strata"]["full_fit" if full_fit else "truncated"] = {
                "examples": len(indices), "api_recall_difference": mean(scores[arm][i]["API_recall"] - scores[baseline][i]["API_recall"] for i in indices) if indices else None}
        report["contrasts"][f"{arm}_minus_{baseline}"] = result
    report["physical_requests"] = {"saved": len(records), "successful": sum(r["status"] == "success" for r in records.values()),
        "prompt_tokens": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in records.values()),
        "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in records.values()),
        "latency_seconds": sum(r.get("latency_seconds", 0) for r in records.values()),
        "known_cost_usd": sum(r.get("cost_micro_usd") or 0 for r in records.values()) / 1e6,
        "usage_missing": sum(r.get("status") == "success" and not r.get("usage") for r in records.values())}
    save(run / "per_example_scores.json", {a: list(v.values()) for a, v in scores.items()})
    save(run / "comparison.json", report)
    with (run / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("arm", "matched_examples", "completed", "ChrF", "API_recall"))
        writer.writeheader()
        for arm, result in report["arms"].items():
            writer.writerow({"arm": arm, "matched_examples": len(common), "completed": result["completed"], **result["metrics"]})
    review = []
    for i in sorted(random.Random(42).sample(common, min(10, len(common)))):
        review.append({"index": i, "instruction": rows[i]["instruction"], "repo": rows[i].get("repo_full_name"),
                       "arms": {arm: {"request_artifact": f"requests/{contexts[(arm, i)]['request_id']}.json",
                                      "extra_library_names": scores[arm][i]["extra_library_names"],
                                      "unlisted_names": scores[arm][i]["unlisted_names"],
                                      "reference_absent_from_hints": scores[arm][i]["reference_absent_from_hints"],
                                      "reference_in_hints_but_unused": scores[arm][i]["reference_in_hints_but_unused"]} for arm in ARMS}})
    save(run / "review_cases.json", review)
    lines = ["# API-list context experiment", "", f"Matched examples: {len(common)}/{len(rows)}; full split: {settings['full_split']}.", "",
             "| Arm | ChrF | API Recall |", "| --- | ---: | ---: |"]
    for arm, result in report["arms"].items():
        vals = [f"{v:.6f}" if v is not None else "N/A" for v in result["metrics"].values()]
        lines.append(f"| {arm} | {' | '.join(vals)} |")
    lines += ["", "| Contrast | Primary | API Recall difference [95% CI] |", "| --- | --- | --- |"]
    for name, result in report["contrasts"].items():
        stats = result["metrics"]["API_recall"]
        cell = f"{stats['difference']:+.6f} [{stats['ci95'][0]:+.6f}, {stats['ci95'][1]:+.6f}]" if stats else "N/A"
        lines.append(f"| {name} | {result['primary']} | {cell} |")
    lines += ["", f"Bootstrap unit: {report['bootstrap_unit']}. Intervals are descriptive, unadjusted for multiplicity.",
              "A null result is not equivalence. Read extra-call diagnostics, repeat variation and context strata in comparison.json.",
              "Calls outside the inventory are not automatically hallucinations. No execution correctness is measured."]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4.1-2025-04-14")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--dataset-file", type=Path, help="Local JSONL or Arrow test data")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--context-window", type=int, default=128000)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--safety-tokens", type=int, default=1024)
    parser.add_argument("--small-budget", type=int, default=8192)
    parser.add_argument("--full-budget", type=int, default=65536)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--tokenizer", choices=("tiktoken", "bytes"), default="tiktoken")
    parser.add_argument("--output-dir", type=Path, default=Path("long_context/results"))
    parser.add_argument("--run-dir", type=Path, help="Evaluate a saved run directly, preserving its generation identity")
    parser.add_argument("--ledger", type=Path, default=Path(".cache/long_context_spending.sqlite"))
    parser.add_argument("--spend-limit-usd", type=float)
    parser.add_argument("--input-usd-per-million", type=float, default=2.)
    parser.add_argument("--output-usd-per-million", type=float, default=8.)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--generate", action="store_true")
    modes.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.run_dir and not args.evaluate_only:
        parser.error("--run-dir requires --evaluate-only")
    if args.limit is not None and args.limit < 1 or args.draws < 100:
        parser.error("Positive limit and at least 100 bootstrap draws required")
    if any(not math.isfinite(v) or v <= 0 for v in (args.input_usd_per_million, args.output_usd_per_million)):
        parser.error("Prices must be finite and positive")
    if args.generate and (args.spend_limit_usd is None or not math.isfinite(args.spend_limit_usd) or args.spend_limit_usd <= 0):
        parser.error("--generate requires an explicit positive --spend-limit-usd")
    from datasets import Dataset, load_dataset
    if args.dataset_file:
        rows = list(Dataset.from_file(str(args.dataset_file))) if args.dataset_file.suffix == ".arrow" else [json.loads(s) for s in args.dataset_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    else:
        rows = list(load_dataset(DATASET, split="test", revision=args.revision))
    if args.run_dir:
        original = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
        contexts = {}
        count_rows = sum(1 for _ in (args.run_dir / "arms" / "plain").glob("*.json"))
        rows = rows[:count_rows]
        if digest(rows) != original["rows_sha256"]:
            parser.error("Dataset does not match saved run; provide its recorded revision or local data")
        for arm in ARMS:
            for index in range(len(rows)):
                context = read_record(args.run_dir / "arms" / arm / f"{index}.json")
                if context["arm"] != arm or context["index"] != index:
                    raise ValueError("Saved context identity mismatch")
                contexts[(arm, index)] = context
        analyze(args.run_dir, rows, contexts, original["settings"], draws=args.draws)
        save(args.run_dir / "evaluation_environment.json", {
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dataset_rows_sha256": digest(rows), "draws": args.draws})
        print(f"Saved evaluation: {args.run_dir / 'REPORT.md'}")
        return
    total = len(rows)
    dataset_sha = digest(rows)
    if not args.dataset_file:
        audit_path = Path("long_context/audit/summary.json")
        if not audit_path.exists():
            parser.error("Run retrieval_recall.py first to create the dataset audit")
        audited = json.loads(audit_path.read_text(encoding="utf-8"))
        if audited["dataset_sha256"] != dataset_sha or audited["dataset_revision"] != args.revision:
            parser.error("Dataset differs from audit; run retrieval_recall.py with this revision first")
        if not audited.get("repository_identity_consistent"):
            parser.error("Repository grouping identity has not been validated")
    rows = rows[:args.limit] if args.limit else rows
    if not rows:
        parser.error("Empty dataset")
    if args.tokenizer == "bytes":
        count, tokenizer = lambda s: len(s.encode("utf-8")), "utf8-byte-upper-bound"
    else:
        import tiktoken
        from ..context.token_counter import EvidenceCounter
        encoding = tiktoken.encoding_for_model(args.model)
        count = EvidenceCounter(encoding)
        tokenizer = {"encoding": encoding.name, "version": tiktoken.__version__}
    settings = {key: getattr(args, key) for key in ("model", "context_window", "max_output_tokens", "safety_tokens", "small_budget", "full_budget", "random_seed", "input_usd_per_million", "output_usd_per_million")}
    settings["full_split"] = len(rows) == total
    root = Path(__file__).resolve().parents[2]
    files = (Path(__file__), root / "src/context/long_context.py", root / "src/context/token_counter.py", root / "src/models/utils.py", root / "src/evaluation/spending.py", root / "src/evaluation/rag_analysis.py", root / "src/metrics/chrf.py", root / "src/metrics/overlap.py", root / "src/context/parsed_file.py", root / "src/context/parser.py")
    packages = {}
    for package in ("datasets", "numpy", "rank-bm25", "tiktoken", "openai", "sacrebleu", "tree-sitter", "tree-sitter-python"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    manifest = {"protocol": 2, "dataset": DATASET if not args.dataset_file else "local-test-data", "revision": args.revision if not args.dataset_file else None,
                "rows_sha256": digest(rows), "settings": settings, "tokenizer": tokenizer, "packages": packages,
                "source_hashes": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                "arms": ARMS, "primary_contrasts": PRIMARY, "price_source": "https://developers.openai.com/api/docs/models/gpt-4.1", "prices_checked": "2026-11-19"}
    experiment = digest(manifest)
    run = args.output_dir / experiment[:20]
    contexts, requests, estimates, audit = prepare(rows, settings, count, experiment)
    save(run / "manifest.json", manifest)
    for (arm, index), context in contexts.items():
        save(run / "arms" / arm / f"{index}.json", sealed(context))
    estimate = {"logical_cases": len(contexts), "unique_requests_including_repeat": len(requests),
                "deduplicated_cases": len(contexts) - len(requests),
                "estimated_input_tokens": sum(v["estimated_input_tokens"] for v in estimates),
                "estimated_max_output_tokens": len(requests) * settings["max_output_tokens"],
                "estimated_cost_with_max_outputs_usd": sum(v["estimated_max_cost_micro_usd"] for v in estimates) / 1e6,
                "conservative_reservation_total_usd": sum(v["reservation_micro_usd"] for v in estimates) / 1e6,
                "largest_single_request_reservation_usd": max(v["reservation_micro_usd"] for v in estimates) / 1e6,
                "note": "Uncached list-price estimate. Ledger reserves byte-based input bound plus framing margin per request; settles actual usage. Include smoke costs through the same ledger."}
    save(run / "preparation.json", {"estimate": estimate, "per_request_estimates": estimates, "audit": audit})
    print(f"Run directory: {run}")
    print(json.dumps(estimate, indent=2))
    if args.generate:
        from openai import OpenAI
        ledger = SpendingLedger(args.ledger, args.spend_limit_usd)
        try:
            ledger_identity_path = run / "ledger_identity.json"
            if ledger_identity_path.exists() and json.loads(ledger_identity_path.read_text())["id"] != ledger.identity:
                raise ValueError("This run is bound to another spending ledger; restore the original ledger")
            save(ledger_identity_path, {"id": ledger.identity})
            execute(run, requests, estimates, ledger, lambda: OpenAI(max_retries=0, timeout=120), args.model)
        finally:
            save(run / "ledger_summary.json", ledger.summary())
            ledger.close()
    if args.generate or args.evaluate_only:
        report = analyze(run, rows, contexts, settings, draws=args.draws)
        print(f"Scored {len(report['common_indices'])}/{len(rows)} matched examples. Report: {run / 'REPORT.md'}")
    else:
        print("Prepared only. No model or embedding calls. Review estimate before choosing a spending cap.")


if __name__ == "__main__":
    main()
