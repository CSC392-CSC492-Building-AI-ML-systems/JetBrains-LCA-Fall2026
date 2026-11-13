"""Verify and export only the non-RAG predictions from a completed local run.

This re-scores saved predictions; it does not generate new model answers or run
project code, GitHub Actions, or retrieval. Missing cases are an error.
"""

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT.parent / "ci-builds-repair-benchmark"
sys.path.insert(0, str(ROOT.parent / "rag"))
from run_benchmark import digest, save_json
from paper_mistral_baseline import make_prompt, parse_response
from evaluation import apply_edits, diff_blocks, prediction_blocks, score


def verify_prediction(prediction, metric, row, files, run, seed, snapshot_hash):
    payload = {k: v for k, v in prediction.items() if k != "payload_sha256"}
    checksum = digest(payload)
    if prediction.get("payload_sha256") != checksum or metric["prediction_sha256"] != checksum:
        raise ValueError("Prediction/metric checksum mismatch")
    expected = {"run": run, "id": row["id"], "seed": seed,
                "method": "baseline", "snapshot": snapshot_hash}
    if prediction["identity"] != expected:
        raise ValueError("Non-RAG prediction identity mismatch")
    if any(metric[k] != expected[k] for k in ("id", "seed", "method")):
        raise ValueError("Metric identity mismatch")
    if any(metric[k] != prediction[k] for k in ("status", "error")):
        raise ValueError("Prediction/metric status mismatch")
    gold = diff_blocks(row["diff"])
    blocks = prediction_blocks(gold)
    if prediction["packed_chunks"] or prediction["prompt"] != make_prompt({"logs": row["logs"]}, blocks):
        raise ValueError("Prompt is not the unaugmented baseline prompt")
    if prediction["status"] == "applied":
        if prediction["error"] or prediction["finish_reason"] == "length":
            raise ValueError("Invalid applied response")
        if parse_response(prediction["response"], blocks) != prediction["replacements"]:
            raise ValueError("Saved replacements differ from model response")
    elif prediction["status"] != "unapplied" or prediction["replacements"] != [b["old"] for b in blocks]:
        raise ValueError("Unapplied response is not an unchanged fallback")
    predicted = apply_edits(files, blocks, prediction["replacements"])
    rescored = score(files, gold, predicted, [], [])
    for key, value in rescored.items():
        if key.startswith("retrieval_"):
            continue
        if metric[key] != value:
            raise ValueError(f"Recomputed metric differs: {key}")
    return {k: v for k, v in metric.items() if not k.startswith("retrieval_")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--repo-cache", type=Path, default=BENCHMARK / ".venv/rag-repos")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    from datasets import Dataset
    from git import Repo

    run = args.run_dir.resolve()
    manifest = json.loads((run / "run_metadata.json").read_text())
    if run.name != digest(manifest)[:20]:
        raise ValueError("Run manifest identity mismatch")
    if hashlib.sha256(args.dataset_file.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Dataset checksum mismatch")
    for name, checksum in manifest["source_hashes"].items():
        source = (BENCHMARK if name == "paper_mistral_baseline.py" else ROOT.parent / "rag") / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Run implementation changed: {name}")
    rows = {r["id"]: r for r in Dataset.from_file(str(args.dataset_file))}
    metrics = []
    for identifier in manifest["ids"]:
        row = rows[identifier]
        provenance = json.loads((run / "retrieval" / f"{identifier}.json").read_text())["provenance"]
        if provenance["sha_fail"] != row["sha_fail"]:
            raise ValueError("Source revision mismatch")
        # Read only target blobs from the exact failing commit; no checkout or execution.
        with Repo(args.repo_cache / f"{row['repo_owner']}__{row['repo_name']}.git") as repo:
            commit = repo.commit(row["sha_fail"])
            files = {}
            for block in diff_blocks(row["diff"]):
                path = block["path"]
                try:
                    blob = commit.tree / path
                except KeyError:
                    if block["count"]:
                        raise ValueError(f"Missing original file: {path}")
                else:
                    files[path] = blob.data_stream.read().decode("utf-8")
        for seed in manifest["seeds"]:
            folder = run / "examples" / str(identifier) / f"seed_{seed}" / "baseline"
            prediction = json.loads((folder / "prediction.json").read_text())
            metric = json.loads((folder / "metrics.json").read_text())
            metrics.append(verify_prediction(prediction, metric, row, files, run.name,
                                             seed, provenance["files_sha256"]))
        print(f"Verified baseline id={identifier}: {len(manifest['seeds'])} seeds", flush=True)
    per_seed = []
    for seed in manifest["seeds"]:
        group = [m for m in metrics if m["seed"] == seed]
        per_seed.append({"seed": seed, "evaluated": len(group),
                         "applied": sum(m["status"] == "applied" for m in group),
                         "exact_reference_matches": sum(m["patch_exact_match"] for m in group),
                         "edit_token_f1": statistics.mean(m["edit_token_f1"] for m in group),
                         "unapplied_reasons": dict(Counter(m["error"] for m in group if m["error"]))})
    result = {"scope": "Verified saved non-RAG predictions; offline proxies, NOT CI Pass@1",
              "source_run": run.name, "fresh_generation": False, "complete": True,
              "expected_predictions": len(manifest["ids"]) * len(manifest["seeds"]),
              "verified_predictions": len(metrics), "retrieved_chunks_in_prompts": 0,
              "per_seed": per_seed,
              "mean_edit_token_f1": statistics.mean(s["edit_token_f1"] for s in per_seed),
              "std_edit_token_f1": statistics.pstdev(s["edit_token_f1"] for s in per_seed),
              "manifest": manifest, "per_example": metrics}
    save_json(args.output_dir / "metrics.json", result)
    columns = ["id", "seed", "status", "difficulty", "patch_exact_match", "edit_token_f1",
               "nonempty_patch", "syntax_eligible_files", "syntax_regression_files", "error",
               "prediction_sha256"]
    with (args.output_dir / "per_example_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps({k: result[k] for k in ("verified_predictions", "retrieved_chunks_in_prompts",
                                           "per_seed", "mean_edit_token_f1", "std_edit_token_f1")}, indent=2))


if __name__ == "__main__":
    main()
