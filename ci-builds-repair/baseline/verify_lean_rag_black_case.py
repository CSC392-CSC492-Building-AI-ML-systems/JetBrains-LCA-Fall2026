"""Replay W&B case 119's pinned Black check on the affected Python file."""

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
import tomllib

import run_qwen_lean_rag as lean

pilot = lean.pilot
CASE = 119
FILE = "wandb/sdk/data_types/image.py"
DEPS = pilot.BENCHMARK / ".venv/local-black-check"
RUN = pilot.BENCHMARK / "out/qwen-lean-rag/bf64963b600ed42de523"

WORKER = """
import importlib.metadata as md
import json, platform, sys
sys.path.insert(0, sys.argv[1])
import black
r = json.load(sys.stdin)
style = r["style"]
mode = black.Mode(line_length=style["line-length"],
                  target_versions={black.TargetVersion[v.upper()] for v in style["target-version"]})
out = {}
for key, text in r["sources"].items():
    try:
        out[key] = {"formatted": black.format_file_contents(text, fast=False, mode=mode), "error": None}
    except black.NothingChanged:
        out[key] = {"formatted": text, "error": None}
    except (black.InvalidInput, SyntaxError, ValueError, AssertionError) as exc:
        out[key] = {"formatted": None, "error": type(exc).__name__}
print(json.dumps({"python": platform.python_version(), "black": md.version("black"), "out": out}))
"""


def changes(lines):
    return [line for line in lines if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--formatter-python", default=sys.executable)
    args = parser.parse_args()
    from datasets import Dataset
    from git import Repo

    manifest = pilot.read_json(RUN / "run_metadata.json")
    if RUN.name != pilot.digest(manifest)[:20]:
        raise ValueError("Run identity mismatch")
    arrow, = [p for p in (pilot.BENCHMARK / ".venv/dataset-cache").rglob("*test.arrow")
              if pilot.file_sha(p) == manifest["dataset_sha256"]]
    row, = [r for r in Dataset.from_file(str(arrow)) if r["id"] == CASE]
    if row["sha_fail"] != manifest["failing_shas"][str(CASE)] or row["changed_files"] != [FILE]:
        raise ValueError("Unexpected W&B formatter case")
    gold = pilot.diff_blocks(row["diff"])
    blocks = pilot.prediction_blocks(gold)
    files = pilot.target_files(row, blocks, lean.CACHE)
    with Repo(lean.CACHE / f"{row['repo_owner']}__{row['repo_name']}.git") as repo:
        tree = repo.commit(row["sha_fail"]).tree
        config = (tree / "pyproject.toml").data_stream.read().decode()
        hook = (tree / ".pre-commit-config.yaml").data_stream.read().decode()
    if "rev: 23.7.0" not in hook or "id: black-jupyter" not in hook:
        raise ValueError("Expected pinned formatter hook")
    sources = {"failed": files[FILE], "reference": pilot.apply_edits(files, blocks, [b["new"] for b in gold])[FILE]}
    if not args.controls_only:
        for seed in manifest["seeds"]:
            for arm in lean.ARMS:
                answer, _ = pilot.read_checked(RUN / "examples" / str(CASE) / str(seed) / f"{arm}.json")
                if any(answer["identity"][k] != v for k, v in
                       {"run": RUN.name, "id": CASE, "seed": seed, "arm": arm}.items()):
                    raise ValueError("Saved answer identity mismatch")
                edits, _ = lean.dev.replacements_for(answer, blocks, "legacy_plain")
                sources[f"{arm}/{seed}"] = pilot.apply_edits(files, blocks, edits)[FILE]
    worker = subprocess.run([args.formatter_python, "-I", "-c", WORKER, str(DEPS)],
                            input=json.dumps({"style": tomllib.loads(config)["tool"]["black"], "sources": sources}),
                            capture_output=True, text=True, timeout=90)
    if worker.returncode:
        raise RuntimeError(f"Black worker failed: {worker.stderr.splitlines()[-1:]}")
    result = json.loads(worker.stdout)
    out = result["out"]
    if result["black"] != "23.7.0" or any(out[k]["error"] for k in ("failed", "reference")):
        raise ValueError("Formatter controls could not be evaluated")
    failed = out["failed"]["formatted"]
    if failed == sources["failed"] or out["reference"]["formatted"] != sources["reference"]:
        raise ValueError("Formatter controls did not reproduce fail/pass")
    formatted_diff = difflib.unified_diff(sources["failed"].splitlines(), failed.splitlines())
    log = lean.ANSI.sub("", row["logs"][0]["log"])
    diff_log = log.partition("All changes made by hooks:")[2].splitlines()
    if changes(formatted_diff) != changes(diff_log):
        raise ValueError("Local formatting edits differ from the failed hook log")
    report = {"scope": "Affected-file Black check; not the full pre-commit job or CI Pass@1",
              "source_run": RUN.name, "case": CASE, "path": FILE, "sha_fail": row["sha_fail"],
              "python": result["python"], "black": result["black"], "failed_changes_match_log": True,
              "results": {
                  key: {"passes_formatter": result["formatted"] == sources[key] and result["error"] is None,
                        "matches_reference_text": sources[key] == sources["reference"],
                        "error": result["error"], "file_sha256": hashlib.sha256(sources[key].encode()).hexdigest()}
                  for key, result in out.items()
              }}
    pilot.save_json(RUN / ("case_119_black_controls.json" if args.controls_only else "case_119_black.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
