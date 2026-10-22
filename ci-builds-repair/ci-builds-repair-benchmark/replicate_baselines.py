"""Submit and collect CI repair reference runs.

Run from this directory with ``poetry run python replicate_baselines.py --help``.
Submission pushes benchmark branches and triggers GitHub Actions. The ``none``
baseline leaves the failing revision unchanged; ``diff`` applies the dataset's
gold repair and is an oracle sanity check, not a deployable repair model.
"""

import argparse
import json
from pathlib import Path


BASELINES = ("none", "diff")


def existing_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("submit", "collect"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--baseline", choices=BASELINES, required=True)
        sub.add_argument("--config", type=existing_file, default=Path(__file__).with_name("config.yaml"))
        sub.add_argument("--output-dir", type=Path, required=True,
                         help="A dedicated directory for this run's JSONL files")
        if command == "submit":
            sub.add_argument("--num-dp", type=int, help="Limit to the first N dataset records")
            sub.add_argument("--ids", type=int, nargs="+", help="Run only these dataset IDs")
            sub.add_argument("--dataset-folder", type=Path, help="Load a local dataset instead of Hugging Face")
        else:
            sub.add_argument("--jobs-file", type=existing_file, required=True,
                             help="jobs_ids_<baseline>.jsonl produced by submit")
    args = parser.parse_args(argv)
    if args.command == "submit" and args.num_dp is not None and args.num_dp < 1:
        parser.error("--num-dp must be positive")
    args.config = existing_file(str(args.config))
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.command == "submit" and args.dataset_folder is not None:
        args.dataset_folder = args.dataset_folder.expanduser().resolve()
        if not args.dataset_folder.is_dir():
            parser.error(f"Dataset folder not found: {args.dataset_folder}")
    if args.command == "collect" and args.jobs_file.parent != args.output_dir:
        parser.error("--jobs-file must be in --output-dir")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Delay third-party imports so --help and argument validation work before setup.
    from benchmark import CIFixBenchmark
    from benhmark_functions import fix_apply_diff, fix_none, get_results

    benchmark = CIFixBenchmark(args.baseline, str(args.config))
    if not benchmark.credentials["token"]:
        raise SystemExit("TOKEN_GH is required in the environment or local .env file")
    benchmark.config.out_folder = str(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = args.output_dir / f"jobs_ids_{args.baseline}.jsonl"
    results_file = args.output_dir / f"jobs_results_{args.baseline}.jsonl"

    if args.command == "submit":
        if jobs_file.exists():
            raise SystemExit(f"Refusing to overwrite existing submissions: {jobs_file}")
        if (args.output_dir / "run_manifest.json").exists():
            raise SystemExit(f"Refusing to overwrite existing run manifest in {args.output_dir}")
        dataset = benchmark.get_dataset(num_dp=args.num_dp,
                                        dataset_folder=str(args.dataset_folder) if args.dataset_folder else None)
        if args.ids is not None:
            selected = set(args.ids)
            dataset = dataset.filter(lambda example: example["id"] in selected)
            missing = selected - set(dataset["id"])
            if missing:
                raise SystemExit(f"Dataset IDs not found: {sorted(missing)}")
        if len(dataset) == 0:
            raise SystemExit("No dataset records selected")
        manifest = {
            "baseline": args.baseline,
            "baseline_kind": "oracle" if args.baseline == "diff" else "no_repair",
            "dataset": benchmark.dataset_id if args.dataset_folder is None else str(args.dataset_folder),
            "dataset_ids": list(dataset["id"]),
            "jobs_file": jobs_file.name,
            "results_file": results_file.name,
        }
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        repair = fix_none if args.baseline == "none" else fix_apply_diff
        print(f"Submitting {len(dataset)} {args.baseline} jobs to GitHub Actions", flush=True)
        benchmark.run_dataset(repair, test_dataset=dataset)
        print(f"Submitted jobs: {jobs_file}")
    else:
        manifest_file = args.output_dir / "run_manifest.json"
        if not manifest_file.is_file():
            raise SystemExit(f"Missing run manifest: {manifest_file}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("baseline") != args.baseline or args.jobs_file.name != manifest.get("jobs_file"):
            raise SystemExit("Baseline or jobs file does not match the run manifest")
        submitted = read_jsonl(args.jobs_file)
        completed = read_jsonl(results_file)
        submitted_by_id = {row["id"]: row for row in submitted}
        if len(submitted_by_id) != len(submitted):
            raise SystemExit("Submitted jobs contain duplicate dataset IDs")
        done = {row["id"] for row in completed}
        if len(done) != len(completed) or not done <= set(submitted_by_id):
            raise SystemExit("Saved results contain duplicate or unknown dataset IDs")
        awaiting, invalid = [], []
        for job in submitted:
            if job["id"] in done:
                continue
            url, conclusion = get_results(job, benchmark.config, benchmark.credentials)
            if conclusion in ("success", "failure"):
                completed.append({**job, "url": url, "conclusion": conclusion})
                done.add(job["id"])
                write_jsonl(results_file, completed)
            elif conclusion == "waiting":
                awaiting.append(job)
            else:
                invalid.append(job)
        write_jsonl(results_file, completed)
        write_jsonl(args.output_dir / f"jobs_awaiting_{args.baseline}.jsonl", awaiting)
        write_jsonl(args.output_dir / f"jobs_invalid_{args.baseline}.jsonl", invalid)
        print(f"Collected {len(completed)}/{len(submitted)} jobs; "
              f"{len(awaiting)} awaiting and {len(invalid)} invalid. Results: {results_file}")
        print("Run evaluate_replication.py on this output directory for coverage and pass rates.")


if __name__ == "__main__":
    main()
