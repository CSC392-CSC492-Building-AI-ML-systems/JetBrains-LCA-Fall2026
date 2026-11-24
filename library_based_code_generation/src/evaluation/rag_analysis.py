"""Evaluate saved numbered LBCG predictions; never generate or call model APIs."""

import argparse
import ast
import csv
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

DATASET = "JetBrains-Research/lca-library-based-code-generation"
ARMS = {
    "plain": "gpt-4.1", "bm25": "bm25/gpt-4.1",
    "dense": "rag_dense_k20/gpt-4.1", "hybrid": "rag_hybrid_k20/gpt-4.1",
    "hybrid_normalized": "rag_hybrid-normalized_k20/gpt-4.1",
}


def portable_path(path):
    """Publish paths relative to LBCG; never expose an external directory."""
    normalized = str(path).replace("\\", "/")
    marker = "library_based_code_generation/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    if normalized.startswith(("src/", "results/", "external/", "rag_evaluation/")) and ".." not in normalized.split("/"):
        return normalized
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()
    except ValueError:
        return "external/" + normalized.rstrip("/").split("/")[-1]


def sanitize_artifact_paths(output):
    """Migrate only path fields in legacy reports, preserving measured results."""
    for filename in ("prediction_hashes.json", "evaluation_manifest.json", "review_cases.json"):
        path = output / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if filename == "prediction_hashes.json":
            data = {portable_path(key): value for key, value in data.items()}
        elif filename == "evaluation_manifest.json":
            data["source_hashes"] = {portable_path(key): value for key, value in data.get("source_hashes", {}).items()}
            if data.get("dataset_file"):
                data["dataset_file"] = portable_path(data["dataset_file"])
        else:
            for row in data:
                for key in ("prediction", "reference_prediction"):
                    row[key] = portable_path(row[key])
        save(path, data)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def extract_code(message):
    # Same extraction as evaluate.py, without importing generation providers.
    for start, end in (("```python", "```"), ("```", "```"), ("<code>", "</code>")):
        if start in message:
            return message.split(start)[1].split(end)[0].strip()
    return message.strip()


def paired_stats(differences, groups, draws=10000, seed=42):
    if not differences:
        return None
    buckets = {}
    for value, group in zip(differences, groups):
        buckets.setdefault(group, []).append(value)
    keys = list(buckets)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = [value for group in rng.choices(keys, k=len(keys)) for value in buckets[group]]
        samples.append(mean(selected))
    samples.sort()
    return {"difference": mean(differences),
            "ci95": [samples[int(.025 * draws)], samples[min(draws - 1, int(.975 * draws))]],
            "wins": sum(d > 0 for d in differences), "losses": sum(d < 0 for d in differences),
            "ties": sum(d == 0 for d in differences), "examples": len(differences), "groups": len(keys)}


def inspect_arm(folder, size, expected_name=None):
    metadata_path = folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    if metadata is not None:
        if metadata.get("split") != "test":
            raise ValueError(f"Expected test split: {folder}")
        if expected_name is not None and metadata.get("name") != expected_name:
            raise ValueError(f"Model identity mismatch: {folder}")
    paths = {}
    for path in folder.glob("*.py"):
        if not path.stem.isdigit():
            continue
        index = int(path.stem)
        if index in paths:
            raise ValueError(f"Duplicate prediction index {index}: {folder}")
        if not 0 <= index < size:
            raise ValueError(f"Unexpected prediction index {index}: {folder}")
        paths[index] = path
    return paths, metadata


def analyze(rows, arms, output, scorer, group_field=None, draws=10000, seed=42):
    if not rows:
        raise ValueError("Empty evaluation dataset")
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    if group_field and any(not isinstance(r.get(group_field), str) or not r[group_field] for r in rows):
        raise ValueError("--group-field must identify a nonempty library/repository on every row")
    if not arms or "plain" not in arms or "bm25" not in arms:
        raise ValueError("Include plain and bm25 reference arms")
    scores, coverage, source_hashes = {}, {}, {}
    for label, folder in arms.items():
        paths, metadata = inspect_arm(folder, len(rows))
        values, invalid = {}, {}
        for index, path in sorted(paths.items()):
            raw = path.read_bytes()
            source_hashes[portable_path(path)] = sha(raw)
            text = raw.decode("utf-8")
            if not text.strip():
                invalid[str(index)] = "empty_prediction"
                continue
            code = extract_code(text)
            try:
                ast.parse(code)
                syntax_valid = True
            except (SyntaxError, ValueError):
                syntax_valid = False
            row = rows[index]
            values[index] = {"index": index, "group": row[group_field] if group_field else str(index),
                             **scorer(code, row), "syntax_valid_diagnostic": syntax_valid,
                             "candidate_api_count": len(set(row.get("project_defined_elements", []))),
                             "reference_api_count": len(row.get("unique_apis", [])),
                             "prediction_sha256": sha(raw), "prediction_chars": len(text)}
        scores[label] = values
        coverage[label] = {"available": len(paths), "scored": len(values), "selected": len(rows),
                           "missing_indices": sorted(set(range(len(rows))) - paths.keys()),
                           "invalid_predictions": invalid, "complete": len(values) == len(rows),
                           "metadata": metadata}
    common = sorted(set.intersection(*(set(v) for v in scores.values())))
    metrics = ("ChrF", "API_recall")
    report = {"selected_examples": len(rows), "common_indices": common,
              "complete": len(common) == len(rows), "coverage": coverage,
              "provenance": "legacy row-order assumption; generation dataset/prompt hashes unavailable",
              "uncertainty_unit": group_field or "individual example; library clustering not accounted for",
              "bootstrap_draws": draws, "bootstrap_seed": seed,
              "means_on_common_examples": {}, "paired_comparisons": {}}
    for label, values in scores.items():
        report["means_on_common_examples"][label] = {
            metric: mean(values[i][metric] for i in common) if common else None for metric in metrics}
        coverage[label]["syntax_valid_on_common"] = sum(values[i]["syntax_valid_diagnostic"] for i in common)
        coverage[label]["zero_api_recall_on_common"] = sum(values[i]["API_recall"] == 0 for i in common)
        saved_metrics = (coverage[label]["metadata"] or {}).get("metrics", {})
        coverage[label]["recomputed_minus_saved_means"] = {
            metric: report["means_on_common_examples"][label][metric] - saved_metrics[metric]["mean"]
            for metric in metrics if len(common) == len(rows) and metric in saved_metrics}
    groups = [rows[i][group_field] if group_field else str(i) for i in common]
    review = []
    for label in scores:
        for reference in ("plain", "bm25"):
            if label == reference:
                continue
            key = f"{label}_minus_{reference}"
            report["paired_comparisons"][key] = {}
            for metric in metrics:
                differences = [scores[label][i][metric] - scores[reference][i][metric] for i in common]
                report["paired_comparisons"][key][metric] = paired_stats(differences, groups, draws, seed)
            ordered = sorted(common, key=lambda i: scores[label][i]["API_recall"] - scores[reference][i]["API_recall"])
            for direction, indices in (("largest_losses", ordered[:5]), ("largest_gains", ordered[-5:][::-1])):
                for i in indices:
                    review.append({"comparison": key, "review_selection": direction, "index": i,
                                   "instruction": rows[i]["instruction"],
                                   "api_recall_difference": scores[label][i]["API_recall"] - scores[reference][i]["API_recall"],
                                   "prediction": portable_path(arms[label] / f"{i}.py"),
                                   "reference_prediction": portable_path(arms[reference] / f"{i}.py")})
    output.mkdir(parents=True, exist_ok=True)
    save(output / "metrics.json", report)
    save(output / "per_example_metrics.json", {label: list(values.values()) for label, values in scores.items()})
    save(output / "review_cases.json", review)
    save(output / "prediction_hashes.json", source_hashes)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "common_examples", *metrics])
        writer.writeheader()
        for label, values in report["means_on_common_examples"].items():
            writer.writerow({"arm": label, "common_examples": len(common), **values})
    lines = ["# LBCG saved RAG evaluation", "",
             f"Common scored examples: {len(common)}/{len(rows)}. Complete: {report['complete']}.", "",
             "Legacy provenance: row order is assumed; generation dataset and prompt identity cannot be verified.", "",
             "| Arm | ChrF | API Recall |", "| --- | ---: | ---: |"]
    for label, values in report["means_on_common_examples"].items():
        cells = [f"{values[m]:.6f}" if values[m] is not None else "N/A" for m in metrics]
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines += ["", "| Paired comparison | Metric | Difference | 95% CI | Wins/losses/ties |",
              "| --- | --- | ---: | --- | --- |"]
    for pair, values in report["paired_comparisons"].items():
        for metric, result in values.items():
            if result:
                lines.append(f"| {pair} | {metric} | {result['difference']:+.6f} | "
                             f"[{result['ci95'][0]:+.6f}, {result['ci95'][1]:+.6f}] | "
                             f"{result['wins']}/{result['losses']}/{result['ties']} |")
    lines += ["", f"Bootstrap unit: {report['uncertainty_unit']}. Intervals are exploratory and unadjusted for multiple comparisons.",
              "", "Syntax validity is a diagnostic, not execution correctness. Review cases are selected for manual inspection, not classified causes.",
              "", "Provider tokens, latency, truncation and cost are unavailable in the legacy prediction artifacts."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("rag_evaluation"))
    parser.add_argument("--dataset-file", type=Path, help="JSONL or saved HF Arrow test data")
    parser.add_argument("--revision", help="Dataset Hub revision; use the generation revision when known")
    parser.add_argument("--group-field", help="Dataset field identifying shared libraries/repositories for clustered bootstrap")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sanitize-paths-only", action="store_true",
                        help="Remove machine-specific paths from existing reports without rescoring")
    args = parser.parse_args()
    if args.sanitize_paths_only:
        sanitize_artifact_paths(args.output_dir)
        print("Converted artifact paths to portable paths; scores unchanged.")
        return
    if args.draws < 100:
        parser.error("--draws must be at least 100")
    from datasets import Dataset, load_dataset
    if args.dataset_file:
        if args.dataset_file.suffix == ".arrow":
            rows = list(Dataset.from_file(str(args.dataset_file)))
        else:
            rows = [json.loads(line) for line in args.dataset_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = list(load_dataset(DATASET, split="test", revision=args.revision))
    from ..metrics.chrf import ChrF
    from ..metrics.overlap import Overlap
    chrf, recall = ChrF(), Overlap()

    def scorer(code, row):
        return {"ChrF": float(chrf.score(code, row["clean_reference"], row["unique_apis"])),
                "API_recall": float(recall.score(code, row["clean_reference"], row["unique_apis"]))}

    arms = {label: args.results_dir / folder for label, folder in ARMS.items()}
    for label, folder in ARMS.items():
        inspect_arm(arms[label], len(rows), folder)
    report = analyze(rows, arms, args.output_dir, scorer, args.group_field, args.draws, args.seed)
    save(args.output_dir / "evaluation_manifest.json", {
        "dataset": DATASET, "revision_requested": args.revision,
        "dataset_file": portable_path(args.dataset_file) if args.dataset_file else None,
        "dataset_content_sha256": sha(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()),
        "source_hashes": {portable_path(p): sha(p.read_bytes()) for p in (
            Path(__file__), Path(__file__).parents[1] / "metrics/chrf.py",
            Path(__file__).parents[1] / "metrics/overlap.py",
            Path(__file__).parents[1] / "context/parsed_file.py",
            Path(__file__).parents[1] / "context/parser.py")}})
    print(f"Scored {len(report['common_indices'])}/{len(rows)} matched examples. Report: {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
