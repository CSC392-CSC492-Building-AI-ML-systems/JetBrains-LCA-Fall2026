"""Verify archived experiment signatures with the exact pre-cleanup source."""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CI = Path("ci-builds-repair")
BENCHMARK = CI / "ci-builds-repair-benchmark"
ARCHIVE = "0915c88"
MISTRAL_RUNS = ("2d9349bb16247814246b", "88a9149b9da105e57f04")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", choices=("all", "mistral", "qwen"), default="all", nargs="?")
    args = parser.parse_args()
    local_benchmark = ROOT / BENCHMARK
    for directory in (".venv", "out"):
        if not (local_benchmark / directory).is_dir():
            raise FileNotFoundError(f"Historical audit needs cached {local_benchmark / directory}")

    with tempfile.TemporaryDirectory(prefix="lca-historical-audit-") as work:
        snapshot = Path(work)
        archive = snapshot / "source.tar"
        subprocess.run(
            ["git", "-C", str(ROOT), "archive", "-o", str(archive), ARCHIVE, str(CI),
             "project_level_code_completion/rag"],
            check=True,
        )
        with tarfile.open(archive) as source:
            source.extractall(snapshot, filter="data")
        for directory in (".venv", "out"):
            (snapshot / BENCHMARK / directory).symlink_to(local_benchmark / directory, target_is_directory=True)

        def audit(script, *extra):
            subprocess.run([sys.executable, str(CI / "baseline" / script), *map(str, extra)],
                           cwd=snapshot, check=True)

        if args.runs in ("all", "mistral"):
            candidates = {}
            for path in (local_benchmark / ".venv/dataset-cache").rglob("*test.arrow"):
                with path.open("rb") as stream:
                    candidates[hashlib.file_digest(stream, "sha256").hexdigest()] = path
            for run in MISTRAL_RUNS:
                folder = local_benchmark / "out/rag-local" / run
                manifest = json.loads((folder / "run_metadata.json").read_text())
                audit("verify_completed_run.py", "--run-dir", folder,
                      "--dataset-file", candidates[manifest["dataset_sha256"]],
                      "--output-dir", snapshot / "audit" / run)
        if args.runs in ("all", "qwen"):
            for script in ("run_qwen_ci_dev.py", "run_qwen_lean_rag.py"):
                audit(script, "--audit-only")


if __name__ == "__main__":
    main()
