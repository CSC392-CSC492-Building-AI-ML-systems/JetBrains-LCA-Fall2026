r"""Run TF-IDF + NLTK on all 50 Python test examples.

This file lives in lca-baselines/bug_localization/baseline_eval.
Run from lca-baselines:
  .\.venv\Scripts\python.exe .\bug_localization\baseline_eval\run_all_tfidf.py

Uses the original classes from commit 977d200d2c2c5dc825190c5293db8e25057ff698.
Preserves the archived baseline's test-directory filter AND unfiltered
dataset labels. A correct file outside the candidates is a missed prediction,
not a failed execution. UTF-8 Git-blob reading remains a portability change;
this is a Python-subset run, not verified paper-score reproduction.
Rerunning resumes matching successful results and retries failed examples.
"""

import ast
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
from importlib.metadata import version
from pathlib import Path


# Source code, repository clones, and outputs live in bug_localization.
ROOT = Path(__file__).resolve().parents[1]
DATASET = "JetBrains-Research/lca-bug-localization"
REFERENCE_COMMIT = "977d200d2c2c5dc825190c5293db8e25057ff698"


def is_test_file(path):
    """Match the archived baseline's substring rule exactly."""
    return any(part in path.lower() for part in ("test/", "tests/"))


def get_labels(value):
    labels = ast.literal_eval(value) if isinstance(value, str) else value
    if (not isinstance(labels, list) or not labels
            or not all(isinstance(path, str) and path for path in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("Expected a nonempty list of unique correct file paths")
    return labels


def save_json(path, value):
    """Replace a JSON file only after its complete replacement is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def digest(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_repository(example):
    from git import Repo

    owner, name = example["repo_owner"], example["repo_name"]
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in (owner, name)):
        raise ValueError("Unexpected GitHub repository name")
    repositories = ROOT / "repos"
    repositories.mkdir(exist_ok=True)
    destination = repositories / f"{owner}__{name}"

    if not destination.exists():
        print(f"  Downloading {owner}/{name} (including Git history)...", flush=True)
        # An interrupted clone cannot leave a half-created final repository.
        with tempfile.TemporaryDirectory(prefix="clone_", dir=repositories) as work:
            temporary = Path(work) / "repository"
            subprocess.run(
                ["git", "clone", "--no-checkout",
                 f"https://github.com/{owner}/{name}.git", str(temporary)],
                check=True,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            temporary.rename(destination)

    if not (destination / ".git").is_dir():
        raise ValueError(f"Existing directory is not a Git clone: {destination}")

    repository = Repo(destination)
    try:
        repository.commit(example["base_sha"])
    except Exception as error:
        # Try obtaining an old commit that is absent from an existing clone.
        print(f"  Fetching missing commit {example['base_sha']}...", flush=True)
        try:
            subprocess.run(
                ["git", "-C", str(destination), "fetch",
                 f"https://github.com/{owner}/{name}.git", example["base_sha"]],
                check=True,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            repository.commit(example["base_sha"])
        except Exception:
            repository.close()
            raise RuntimeError(
                f"Cannot load base_sha {example['base_sha']}; "
                "the benchmark's archived repository may be needed."
            ) from error
    return repository


def read_content(repository, commit_sha):
    """Read the buggy snapshot without checking out or executing source code."""
    contents = {}
    for item in repository.commit(commit_sha).tree.traverse():
        if item.type != "blob" or not item.path.endswith(".py"):
            continue
        if is_test_file(item.path):
            continue
        try:
            contents[item.path] = item.data_stream.read().decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Non-UTF-8 source file: {item.path}") from error
    if not contents:
        raise ValueError("No candidate Python files found")
    return contents


def run_examples(dataset, baseline, output_dir, manifest):
    run_signature = digest(manifest)
    save_json(output_dir / "run_metadata.json", manifest)
    successes, failures = [], []
    summary = None

    for index, example in enumerate(dataset):
        identity = {
            key: example[key] for key in (
                "repo_owner", "repo_name", "base_sha", "issue_url",
                "issue_title", "issue_body", "changed_files",
            )
        }
        signature = digest({"run": run_signature, "example": identity})
        result_path = output_dir / "examples" / f"{index:03d}_{signature[:12]}.json"
        repository_name = f"{example['repo_owner']}__{example['repo_name']}"
        print(f"\n[{index + 1}/{len(dataset)}] {repository_name}", flush=True)

        result = None
        if result_path.exists():
            try:
                cached = json.loads(result_path.read_text(encoding="utf-8"))
                if cached.get("signature") == signature and cached.get("status") == "success":
                    result = cached
                    print("  Reusing saved result.", flush=True)
            except (ValueError, OSError):
                pass

        if result is None:
            started = time.perf_counter()
            result = {
                "signature": signature, "example_index": index,
                "repository": repository_name, "base_sha": example["base_sha"],
                "issue_url": example["issue_url"],
            }
            try:
                repository = get_repository(example)
                try:
                    contents = read_content(repository, example["base_sha"])
                finally:
                    repository.close()

                query = example["issue_title"] + "\n" + example["issue_body"]
                print(f"  Ranking {len(contents)} Python files...", flush=True)
                predicted = baseline.localize_bugs(query, contents)
                files = [str(path) for path in predicted["final_files"]]
                scores = [float(score) for score in predicted["rank_scores"]]
                if (len(files) != len(contents) or len(scores) != len(files)
                        or set(files) != set(contents)
                        or not all(math.isfinite(score) for score in scores)):
                    raise ValueError("Baseline returned an invalid/incomplete ranking")

                # Labels are consulted AFTER prediction, never passed to the model.
                labels = get_labels(example["changed_files"])
                missing = [path for path in labels if path not in contents]
                # run_baseline.py retains dp['changed_files'] even though its
                # HFDataSource excludes tests from repo_content. Preserve those
                # labels in metric denominators rather than dropping the issue.
                ranks = {path: rank for rank, path in enumerate(files, 1)}

                result.update({
                    "status": "success", "candidate_count": len(contents),
                    "correct_files": labels,
                    "correct_file_ranks": {path: ranks.get(path) for path in labels},
                    "unranked_correct_files": missing,
                    "excluded_test_correct_files": [path for path in missing if is_test_file(path)],
                    "other_unranked_correct_files": [path for path in missing if not is_test_file(path)],
                    "label_policy": "Raw dataset changed_files; unreachable labels count as misses",
                    "ranking": [
                        {"rank": rank, "file": path, "score": score}
                        for rank, (path, score) in enumerate(zip(files, scores), 1)
                    ],
                })
            except Exception as error:
                result.update({
                    "status": "failed", "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                })
            result["elapsed_seconds"] = time.perf_counter() - started
            save_json(result_path, result)

        if result["status"] == "success":
            successes.append(str(result_path.relative_to(output_dir)))
            print("  Top prediction:", result["ranking"][0]["file"], flush=True)
            print("  Correct file ranks:", result["correct_file_ranks"], flush=True)
            if result.get("unranked_correct_files"):
                print("  Unranked correct files (count as misses):",
                      result["unranked_correct_files"], flush=True)
        else:
            failures.append({"index": index, "issue_url": example["issue_url"], "error": result["error"]})
            print("  FAILED:", result["error"], flush=True)

        summary = {
            "baseline": "TF-IDF + NLTK", "configuration": "py", "split": "test",
            "total_examples": len(dataset), "processed": index + 1,
            "successful": len(successes), "failed": len(failures),
            "remaining": len(dataset) - index - 1,
            "all_examples_successful": len(successes) == len(dataset),
            "label_policy": "Raw dataset changed_files; unreachable labels count as misses",
            "result_files": successes, "failures": failures,
            "note": "Counts only. Aggregate benchmark metrics have not been calculated.",
        }
        save_json(output_dir / "summary.json", summary)
    return summary


def main():
    # Make src importable when running this script directly from any directory.
    sys.path.insert(0, str(ROOT))
    from datasets import load_dataset
    import nltk
    from src.baselines.backbones.emb.tfidf_emb_backbone import TfIdfEmbBackbone
    from src.baselines.backbones.emb.tokenizers.nltk_tokenizer import NltkTokenizer
    from src.baselines.backbones.emb.rankers.cosine_distance_ranker import CosineDistanceRanker

    # Fetch the additional tokenizer data needed by newer NLTK versions.
    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet"):
        nltk.download(resource, quiet=True, raise_on_error=True)

    print("Loading Python test split...", flush=True)
    dataset = load_dataset(DATASET, "py", split="test")
    if len(dataset) != 50:
        raise ValueError(f"Expected 50 Python test examples, received {len(dataset)}")

    relevant_files = [
        "src/baselines/backbones/emb/tfidf_emb_backbone.py",
        "src/baselines/backbones/emb/tokenizers/nltk_tokenizer.py",
        "src/baselines/backbones/emb/rankers/cosine_distance_ranker.py",
        "src/baselines/utils/embed_utils.py",
    ]
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "dataset": DATASET, "configuration": "py", "split": "test",
        "dataset_fingerprint": dataset._fingerprint,
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scikit-learn", "nltk", "datasets", "GitPython")},
        "source_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in relevant_files
        },
        "runner_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "snapshot_method": "Git blobs decoded as UTF-8; .py only; exclude test/ and tests/",
        "label_policy": "Raw dataset changed_files; unreachable labels count as misses",
        "baseline": "Original TF-IDF + NLTK classes; fit per issue and its repository",
    }
    # Different code/data/package versions get separate folders automatically.
    output_dir = ROOT / "outputs" / "tfidf_python50" / digest(manifest)[:12]
    # The evaluator can select the new run even when older runs are present.
    save_json(output_dir.parent / "latest_run.json", {
        "run_directory": output_dir.name,
        "manifest_digest": digest(manifest),
    })
    baseline = TfIdfEmbBackbone(
        name="tfidf", tokenizer=NltkTokenizer(),
        ranker=CosineDistanceRanker(), pretrained_path=None,
    )
    print("Results folder:", output_dir, flush=True)
    summary = run_examples(dataset, baseline, output_dir, manifest)
    print(f"\nFinished: {summary['successful']}/50 successful; {summary['failed']} failed.")
    print("Summary:", output_dir / "summary.json")
    if summary["failed"]:
        print("Failures are recorded. Fix their causes and rerun to retry them.")
    else:
        print("All 50 rankings saved. Run evaluate_tfidf.py to calculate full-split metrics.")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
