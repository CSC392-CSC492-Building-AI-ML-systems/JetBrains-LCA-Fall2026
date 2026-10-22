"""Score saved CI repair jobs without network access or benchmark dependencies."""

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    if not path.is_file():
        raise ValueError(f"Missing file: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path.name}:{line_number}: {error}") from error
    return rows


def score_run(directory):
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    baseline = manifest["baseline"]
    if baseline not in ("none", "diff"):
        raise ValueError(f"Unknown baseline: {baseline}")
    submitted = read_jsonl(directory / manifest["jobs_file"])
    results = read_jsonl(directory / manifest["results_file"])
    selected_ids = manifest["dataset_ids"]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Duplicate selected dataset IDs")
    submitted_ids = [row["id"] for row in submitted]
    if len(submitted_ids) != len(set(submitted_ids)) or not set(submitted_ids) <= set(selected_ids):
        raise ValueError("Duplicate or unexpected submitted dataset IDs")
    result_ids = [row["id"] for row in results]
    if len(result_ids) != len(set(result_ids)) or not set(result_ids) <= set(submitted_ids):
        raise ValueError("Duplicate or unexpected result dataset IDs")
    submitted_by_id = {row["id"]: row for row in submitted}
    for row in results:
        original = submitted_by_id[row["id"]]
        if row.get("commit") != original.get("commit") or row.get("branch_name") != original.get("branch_name"):
            raise ValueError(f"Result does not match submitted job {row['id']}")
        if row.get("conclusion") not in ("success", "failure"):
            raise ValueError(f"Unexpected conclusion for job {row['id']}: {row.get('conclusion')}")
    counts = Counter(row["conclusion"] for row in results)
    by_difficulty = {}
    for difficulty in sorted({str(row["difficulty"]) for row in results}):
        group = [row for row in results if str(row["difficulty"]) == difficulty]
        passes = sum(row["conclusion"] == "success" for row in group)
        by_difficulty[difficulty] = {
            "completed": len(group), "passed": passes, "pass_rate": passes / len(group)
        }
    return {
        "baseline": baseline,
        "baseline_kind": manifest["baseline_kind"],
        "selected": len(selected_ids),
        "submitted": len(submitted),
        "completed": len(results),
        "not_submitted_ids": sorted(set(selected_ids) - set(submitted_ids)),
        "unresolved_ids": sorted(set(submitted_ids) - set(result_ids)),
        "passed": counts["success"], "failed": counts["failure"],
        "coverage": len(results) / len(selected_ids) if selected_ids else 0,
        "pass_rate_completed": counts["success"] / len(results) if results else None,
        "pass_rate_selected_lower_bound": counts["success"] / len(selected_ids) if selected_ids else None,
        "by_difficulty": by_difficulty,
        "complete": len(results) == len(selected_ids),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory containing run_manifest.json and job JSONL files")
    args = parser.parse_args(argv)
    try:
        summary = score_run(args.output_dir.expanduser().resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    output = args.output_dir / "replication_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{summary['baseline']}: {summary['completed']}/{summary['selected']} completed; "
          f"{summary['passed']} passed; {summary['failed']} failed")
    print(f"Completed-job pass rate: {summary['pass_rate_completed']}")
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
