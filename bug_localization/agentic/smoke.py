"""Offline end-to-end Git -> agent -> saved trace -> existing evaluator."""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from git import Repo

from bug_localization.agentic.evaluate import compare_runs
from bug_localization.agentic.run import run_examples
from bug_localization.agentic.tests.fixtures import GuardedRow, InvestigatingClient, make_repository
from bug_localization.long_context import experiment
from bug_localization.rag.model import ModelSettings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "offline_smoke")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir, prefix="fixture-") as work:
        repository, row = make_repository(Path(work) / "repo")
        with patch.object(experiment, "get_repository", side_effect=lambda _: Repo(repository)):
            summary = run_examples([GuardedRow(row)], 1, args.output_dir / "runs",
                                   ModelSettings(context_window=16000, max_output_tokens=500, tokenizer="bytes"),
                                   window=4, stride=2, client=InvestigatingClient(), dataset_name="offline-fixture")
        report = compare_runs([summary], [row], args.output_dir / "scoring", allow_partial=True)
        result = json.loads((summary.parent / "examples" / "000.json").read_bytes())
        assert result["trace"][1]["action"]["query"] == "resolve_alias"
        assert result["ranking"][0]["file"] == "symbols.py"
        assert report["runs"][0]["metrics"]["MAP_full_ranking"] == .5
        print(f"OFFLINE MOCK ONLY; trace: {summary.parent / 'examples' / '000.json'}")
        print("Discovered resolve_alias -> searched it -> read symbols.py -> submitted scorable ranking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
