"""Replay SkyPilot case 37's formatter on the affected file, without executing it."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import run_qwen_ci_dev as dev

pilot = dev.pilot
FORMATTER = pilot.BENCHMARK / ".venv/local-format-check"
CASE_ID = 37
FILE = "sky/resources.py"

# Run only the pinned formatter and TOML parser in a Python version with lib2to3.
# Project files, including generated edits, are data sent via stdin.
WORKER = """
import importlib.metadata as md
import json, platform, sys
sys.path.insert(0, sys.argv[1])
import toml
from yapf.yapflib.yapf_api import FormatCode
r = json.load(sys.stdin)
style = toml.loads(r["config"])["tool"]["yapf"]
out = {}
for key, source in r["sources"].items():
    diff, changed = FormatCode(source, filename=r["file"], style_config=style, print_diff=True)
    out[key] = {"would_reformat": changed, "diff": diff}
print(json.dumps({"python": platform.python_version(), "packages": {
    p: md.version(p) for p in ("yapf", "toml")}, "results": out}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=pilot.BENCHMARK / "out/qwen-ci-dev/ac114173bcf206333b0a")
    parser.add_argument("--formatter-python", default="/usr/bin/python3")
    args = parser.parse_args()
    from datasets import Dataset
    from git import Repo

    run = args.run.resolve()
    manifest = pilot.read_json(run / "run_metadata.json")
    if run.name != pilot.digest(manifest)[:20] or not pilot.read_json(run / "metrics.json")["complete"]:
        raise ValueError("Expected a complete Qwen development run")
    data, = [p for p in (pilot.BENCHMARK / ".venv/dataset-cache").rglob("*test.arrow")
             if pilot.file_sha(p) == manifest["dataset_sha256"]]
    row, = [r for r in Dataset.from_file(str(data)) if r["id"] == CASE_ID]
    if row["sha_fail"] != manifest["failing_shas"][str(CASE_ID)] or row["changed_files"] != [FILE]:
        raise ValueError("Unexpected formatter case identity")
    gold = pilot.diff_blocks(row["diff"])
    blocks = pilot.prediction_blocks(gold)
    repos = pilot.BENCHMARK / ".venv/rag-repos"
    files = pilot.target_files(row, blocks, repos)
    with Repo(repos / f"{row['repo_owner']}__{row['repo_name']}.git") as repo:
        config = (repo.commit(row["sha_fail"]).tree / "pyproject.toml").data_stream.read().decode("utf-8")
    sources = {"failing_revision": files[FILE],
               "reference_patch": pilot.apply_edits(files, blocks, [b["new"] for b in gold])[FILE]}
    for seed in manifest["seeds"]:
        for arm in dev.ARMS:
            answer, _ = pilot.read_checked(run / "examples" / str(CASE_ID) / str(seed) / f"{arm}.json")
            if answer["identity"] != {"run": run.name, "id": CASE_ID, "seed": seed, "arm": arm,
                                       "sha_fail": row["sha_fail"]}:
                raise ValueError("Prediction identity mismatch")
            edits, error = dev.replacements_for(answer, blocks, arm)
            if error:
                raise ValueError(f"Invalid prediction for case {CASE_ID}: {arm}/{seed}: {error}")
            sources[f"{arm}/seed_{seed}"] = pilot.apply_edits(files, blocks, edits)[FILE]
    subprocess_result = subprocess.run(
        [args.formatter_python, "-I", "-c", WORKER, str(FORMATTER)],
        input=json.dumps({"config": config, "file": FILE, "sources": sources}),
        capture_output=True, text=True, check=True, timeout=90,
    )
    observed = json.loads(subprocess_result.stdout)
    if observed["packages"] != {"yapf": "0.32.0", "toml": "0.10.2"}:
        raise ValueError("Wrong formatter versions")
    failed = observed["results"]["failing_revision"]
    fixed = observed["results"]["reference_patch"]
    if not failed["would_reformat"] or fixed["would_reformat"]:
        raise ValueError("Failed/reference formatter controls did not reproduce")
    def hunks(text):
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("@@ "))
        return [line for line in lines[start:] if not line.startswith("##[error]")]

    if hunks(failed["diff"]) != hunks(row["logs"][0]["log"]):
        raise ValueError("Local formatting hunk differs from recorded failure")
    report = {
        "scope": "Affected-file YAPF check only, not recursive workflow or CI Pass@1",
        "case": CASE_ID, "path": FILE, "sha_fail": row["sha_fail"], "source_run": run.name,
        "failing_hunk_matches_recorded_log": True, "python": observed["python"],
        "packages": observed["packages"], "results": {
            key: {"passes_formatter": not value["would_reformat"],
                  "matches_reference_text": sources[key] == sources["reference_patch"],
                  "file_sha256": hashlib.sha256(sources[key].encode()).hexdigest()}
            for key, value in observed["results"].items()
        },
    }
    pilot.save_json(run / "case_37_yapf.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
