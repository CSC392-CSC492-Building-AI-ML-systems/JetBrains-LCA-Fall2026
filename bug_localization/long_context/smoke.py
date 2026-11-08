"""Create five reproducible offline fixture runs and a comparison table. No API."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

from git import Repo

from bug_localization.long_context import experiment
from bug_localization.long_context.compare import compare_runs
from bug_localization.long_context.run import run_examples
from bug_localization.long_context.tests.fixtures import GuardedRow, MockClient, make_repository
from bug_localization.rag.model import ModelSettings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "offline_smoke")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The temporary clone stays inside the experiment folder; no benchmark clone
    # cache, dataset download or API credentials are required.
    with tempfile.TemporaryDirectory(dir=args.output_dir, prefix="fixture-") as work:
        repo_path, rows = make_repository(Path(work) / "repository")
        paths = []
        with patch.object(experiment, "get_repository", side_effect=lambda _: Repo(repo_path)):
            for mode, budget in (("direct", 80), ("rag", 80), ("direct", 512), ("rag", 512), ("issue_only", 0)):
                paths.append(run_examples([GuardedRow(row) for row in rows], mode, budget,
                                          args.output_dir / "runs",
                                          ModelSettings(context_window=12000, max_output_tokens=500, tokenizer="bytes"),
                                          client=MockClient(), dataset_name="offline-synthetic-fixture"))
        compare_runs(paths, rows, args.output_dir)
    path = args.output_dir / "comparison.md"
    text = path.read_text(encoding="utf-8")
    path.write_text("OFFLINE FIXTURE ONLY: two synthetic issues and mocked responses; these are not LCA benchmark results.\n\n"
                    + text, encoding="utf-8")
    print(path.read_text(encoding="utf-8"))
    print(f"Offline comparison: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
