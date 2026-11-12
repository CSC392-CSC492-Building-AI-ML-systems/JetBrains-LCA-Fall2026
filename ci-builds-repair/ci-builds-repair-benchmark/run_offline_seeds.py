"""Run seeded Mistral predictions in isolated local clones, without GitHub pushes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from datasets import load_dataset

from paper_mistral_baseline import MistralRepair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, required=True, help="Dataset record ID")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--source-repo", type=Path, required=True,
                        help="Local Git clone containing the failing commit")
    parser.add_argument("--model-path", type=Path, default=Path(__file__).parent / ".venv/mistral-model")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or args.temperature < 0:
        parser.error("Seeds must be unique and temperature must be nonnegative")
    if not (args.source_repo / ".git").exists():
        parser.error("--source-repo must be a Git checkout")
    dataset = load_dataset("JetBrains-Research/lca-ci-builds-repair", split="test",
                           cache_dir=str(Path(__file__).parent / ".venv/dataset-cache"))
    matches = [row for row in dataset if row["id"] == args.id]
    if len(matches) != 1:
        parser.error(f"Expected one record with ID {args.id}; found {len(matches)}")
    row = matches[0]
    root = args.output_dir.resolve()
    if root.exists():
        parser.error(f"Output directory already exists: {root}")
    root.mkdir(parents=True)
    model = MistralRepair(args.model_path, seed=args.seeds[0], temperature=args.temperature)
    outcomes = []
    for seed in args.seeds:
        destination = root / f"seed_{seed}"
        clone = destination / "repo"
        destination.mkdir()
        subprocess.run(["git", "clone", "--local", "--no-checkout", str(args.source_repo.resolve()),
                        str(clone)], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "--detach", row["sha_fail"]],
                       check=True, capture_output=True, text=True)
        model.seed = seed
        model(row, str(clone), out_folder=str(destination))
        patch = subprocess.run(["git", "-C", str(clone), "diff", "--binary"],
                               check=True, capture_output=True, text=True).stdout
        (destination / "prediction.patch").write_text(patch, encoding="utf-8")
        generation = json.loads((destination / f"model_generation_{row['id']}.json").read_text(encoding="utf-8"))
        outcomes.append({"seed": seed, "status": generation["status"],
                         "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                         "patch_bytes": len(patch.encode())})
        print(f"seed {seed}: {generation['status']}, {len(patch.encode())} patch bytes", flush=True)
    summary = {"dataset": "JetBrains-Research/lca-ci-builds-repair", "dataset_fingerprint": dataset._fingerprint,
               "dataset_size": len(dataset), "id": args.id, "temperature": args.temperature,
               "outcomes": outcomes,
               "identical_patches": len({item["patch_sha256"] for item in outcomes}) == 1}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Identical patches across seeds: {summary['identical_patches']}")


if __name__ == "__main__":
    main()
