"""Check the dataset's reference repairs against failing snapshots, locally.

This is a gold-diff oracle, not a predictive model or a CI pass-rate evaluator.
It reads cached dataset and Git objects and never checks out or runs project code.
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

from datasets import Dataset


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent / "ci-builds-repair-benchmark"
RAG = HERE.parent / "rag"
sys.path.insert(0, str(RAG))

from evaluation import apply_edits, diff_blocks, edit_tokens, score  # noqa: E402
from run_benchmark import snapshot  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--repo-cache", type=Path, default=BENCHMARK / ".venv/rag-repos")
    parser.add_argument("--output", type=Path, default=BENCHMARK / "out/reference-oracle/summary.json")
    args = parser.parse_args()

    if args.dataset_file is None:
        candidates = sorted((BENCHMARK / ".venv/dataset-cache").rglob("*test.arrow"))
        if len(candidates) != 1:
            parser.error("Pass --dataset-file: need exactly one cached test Arrow file")
        args.dataset_file = candidates[0]

    dataset = Dataset.from_file(str(args.dataset_file))
    rows = []
    for row in dataset:
        files, provenance = snapshot(row, args.repo_cache)
        blocks = diff_blocks(row["diff"])
        reference = apply_edits(files, blocks, [block["new"] for block in blocks])
        metrics = score(files, blocks, reference, [], [])
        token_count = sum(edit_tokens(files, reference).values())
        if not metrics["patch_exact_match"] or (token_count and metrics["edit_token_f1"] != 1.0):
            raise RuntimeError(f"Reference repair did not reproduce dataset diff for ID {row['id']}")
        rows.append({"id": row["id"], "repository": provenance["repository"],
                     "sha_fail": provenance["sha_fail"], "changed_files": sorted(reference),
                     "exact_reference_match": True,
                     "edit_token_f1": metrics["edit_token_f1"] if token_count else None})
        print(f"[{len(rows)}/{len(dataset)}] reference repair matched ID {row['id']}", flush=True)

    defined_f1 = [row["edit_token_f1"] for row in rows if row["edit_token_f1"] is not None]
    result = {"scope": "Dataset gold-diff oracle; not Mistral accuracy or CI Pass@1",
              "dataset_sha256": hashlib.sha256(args.dataset_file.read_bytes()).hexdigest(),
              "dataset_size": len(dataset), "verified": len(rows),
              "exact_reference_matches": len(rows),
              "tokenless_repairs": sum(row["edit_token_f1"] is None for row in rows),
              "mean_edit_token_f1_when_defined": statistics.mean(defined_f1) if defined_f1 else None,
              "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Matched {len(rows)}/{len(dataset)} reference repairs; summary: {args.output}")


if __name__ == "__main__":
    main()
