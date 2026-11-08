"""Retrieve code and a BM25-only full file ranking; no generation or labels.

Example: python -m bug_localization.rag.run_retrieval --limit 1 --evaluate
"""

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bug_localization.rag import bm25_index, prepare_repositories as preparation
from bug_localization.rag import retrieval_tokenizer as tokenizer
from bug_localization.rag.retrieval import (
    PREDICTION_FIELDS, RETRIEVAL_VERSION, digest, make_query, project_example,
    retrieve, validate_prediction,
)


DEFAULT_OUTPUT = preparation.RAG_ROOT / "outputs" / "bm25_retrieval"


@dataclass(frozen=True)
class Settings:
    max_chunks: int = 30
    window: int = 40
    stride: int = 20
    k1: float = 1.2
    b: float = 0.75

    def validate(self):
        preparation.validate_settings(self.window, self.stride)
        bm25_index.validate_settings(self.k1, self.b)
        if type(self.max_chunks) is not int or self.max_chunks < 0:
            raise ValueError("max_chunks must be a nonnegative integer")


def save_json(path, value):
    """Atomically replace one complete JSON artifact, even with concurrent writers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".writing-", dir=path.parent) as work:
        temporary = Path(work) / path.name
        temporary.write_bytes(preparation.json_bytes(value))
        temporary.replace(path)


def load_dataset_input(split="test", dataset_cache_dir=None, dataset_file=None):
    from datasets import Dataset, load_dataset

    if dataset_file is not None:
        return Dataset.from_file(str(dataset_file))
    return load_dataset(preparation.DATASET, "py", split=split,
                        cache_dir=str(dataset_cache_dir) if dataset_cache_dir else None)


def prediction_columns(dataset):
    # Project Arrow columns BEFORE row materialization, so retrieval never even
    # loads changed_files/head_sha/diff fields. Mapping fixtures are read through
    # project_example's explicit allowlist instead.
    if hasattr(dataset, "select_columns"):
        return dataset.select_columns([key for key in PREDICTION_FIELDS if key in dataset.column_names])
    return dataset


def load_matching_index(example, settings, prepared_cache, index_cache, repository_cache_dir=None):
    provenance = {key: example[key] for key in ("repo_owner", "repo_name", "base_sha")}
    owner, name, sha = (provenance[key] for key in ("repo_owner", "repo_name", "base_sha"))
    if (not all(isinstance(p, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", p) for p in (owner, name))
            or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{4,64}", sha)):
        raise ValueError("Invalid repository identity or base_sha")
    repository_name = f"{owner}__{name}"
    snapshot = None
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha):
        # A full SHA already resolved by step 1 needs no Git/network access.
        identity = preparation.cache_identity(repository_name, sha, settings.window, settings.stride)
        snapshot = preparation.load_cache(Path(prepared_cache) / digest(identity), identity)
    if snapshot is None:
        if repository_cache_dir is None:
            repository = preparation.get_repository(provenance)
        else:
            # An optional existing clone cache is read only; no checkout/fetch.
            from git import Repo
            repository = Repo(Path(repository_cache_dir) / repository_name)
        try:
            snapshot = preparation.prepare_snapshot(repository, repository_name, sha,
                                                    prepared_cache, settings.window, settings.stride)
        finally:
            repository.close()
    expected = preparation.cache_identity(repository_name, snapshot.metadata["base_sha"],
                                          settings.window, settings.stride)
    if (not snapshot.metadata["base_sha"].startswith(sha)
            or any(snapshot.metadata[key] != value for key, value in expected.items())):
        raise ValueError("Prepared snapshot provenance differs from requested example")
    index = bm25_index.build_or_load_index(snapshot, index_cache, settings.k1, settings.b)
    identity = bm25_index.cache_identity(snapshot, settings.k1, settings.b)
    metadata = json.loads((index.cache_path / "metadata.json").read_bytes())
    if (any(metadata[key] != value for key, value in identity.items())
            or digest(identity) != index.cache_path.name or index.snapshot is not snapshot):
        raise ValueError("BM25 index identity differs from prepared snapshot/settings")
    return index


def implementation_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ["bug_localization/rag/" + name for name in (
        "retrieval.py", "run_retrieval.py", "prepare_repositories.py",
        "bm25_index.py", "retrieval_tokenizer.py")]
    names.extend(["project_level_code_completion/rag/bm25.py",
                  "bug_localization/baseline_eval/run_all_tfidf.py"])
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _reuse(path, expected_identity, index, settings):
    try:
        result = json.loads(path.read_bytes())
        checksum = result.pop("payload_sha256")
        if (checksum != digest(result) or result["status"] != "success"
                or result["identity"] != expected_identity
                or result["signature"] != digest(expected_identity)):
            return None
        validate_prediction(result, index, settings.max_chunks)
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def run_examples(dataset, output_root=DEFAULT_OUTPUT, indices=None, settings=Settings(),
                 split="test", prepared_cache=preparation.DEFAULT_CACHE,
                 index_cache=bm25_index.DEFAULT_CACHE, repository_cache_dir=None, resume=True):
    """Write label-free per-example artifacts; failures are explicit and retried."""
    settings.validate()
    indices = list(indices if indices is not None else range(min(1, len(dataset))))
    if (not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(dataset) for i in indices)):
        raise ValueError("Require nonempty, distinct in-range example indices")
    manifest = {
        "retrieval_version": RETRIEVAL_VERSION, "dataset": preparation.DATASET,
        "configuration": "py", "split": split,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_total_examples": len(dataset), "example_indices": indices,
        "settings": asdict(settings), "tokenizer": tokenizer.tokenizer_identity(),
        "bm25_scoring_version": bm25_index.SCORING_VERSION,
        "query_policy": "nonblank-issue_title-blank-line-issue_body/v1",
        "file_scoring": "maximum-over-all-indexed-chunks; ties=path/v1",
        "source_hashes": implementation_hashes(),
    }
    run_signature = digest(manifest)
    output_dir = Path(output_root).resolve() / run_signature
    save_json(output_dir / "run_metadata.json", manifest)
    save_json(output_dir.parent / "latest_run.json", {"run_directory": run_signature})
    dataset = prediction_columns(dataset)
    summary = {
        "baseline": "BM25 max-chunk", "configuration": "py", "split": split,
        "run_signature": run_signature, "dataset_total_examples": len(dataset),
        "example_indices": indices, "total_examples": len(indices), "processed": 0,
        "successful": 0, "failed": 0, "remaining": len(indices), "reused": 0,
        "result_files": [], "failures": [],
    }
    for example_index in indices:
        path = output_dir / "examples" / f"{example_index:03d}.json"
        started = time.perf_counter()
        result = {"status": "failed", "example_index": example_index}
        try:
            example = project_example(dataset[example_index])
            query = make_query(example)
            result.update({"example_input": example, "query": query})
            index = load_matching_index(example, settings, prepared_cache, index_cache, repository_cache_dir)
            index_identity = bm25_index.cache_identity(index.snapshot, settings.k1, settings.b)
            identity = {
                "run_signature": run_signature, "example_index": example_index,
                "example_input": example, "query": query,
                "prepared_metadata": index.snapshot.metadata,
                "index_identity": index_identity,
            }
            cached = _reuse(path, identity, index, settings) if resume else None
            if cached is not None:
                result = cached
                summary["reused"] += 1
            else:
                result = {
                    "status": "success", "identity": identity, "signature": digest(identity),
                    "example_index": example_index, "issue_url": example["issue_url"],
                    "repository": index.snapshot.metadata["repository"],
                    "base_sha": index.snapshot.metadata["base_sha"], "query": query,
                    "requested_max_chunks": settings.max_chunks,
                    "prepared_snapshot": {"cache_path": str(index.snapshot.cache_path.resolve()),
                                          "metadata_sha256": index_identity["prepared_metadata_sha256"]},
                    "bm25_index": {"cache_path": str(index.cache_path.resolve()),
                                   "key": digest(index_identity), "identity": index_identity},
                    **retrieve(index, query, settings.max_chunks),
                }
                validate_prediction(result, index, settings.max_chunks)
                result["elapsed_seconds"] = time.perf_counter() - started
                save_json(path, {**result, "payload_sha256": digest(result)})
            summary["result_files"].append(path.relative_to(output_dir).as_posix())
            summary["successful"] += 1
            print(f"[{example_index}] {result['repository']} {result['base_sha']} "
                  f"chunks={len(result['retrieved_chunks'])}/{settings.max_chunks} "
                  f"distinct_files={result['distinct_retrieved_files']} candidates={result['candidate_count']} "
                  f"{'reused' if cached is not None else 'written'}", flush=True)
        except Exception as error:
            result = {"status": "failed", "example_index": example_index,
                      "run_signature": run_signature, "error": f"{type(error).__name__}: {error}",
                      "elapsed_seconds": time.perf_counter() - started}
            save_json(path, {**result, "payload_sha256": digest(result)})
            summary["failed"] += 1
            summary["failures"].append({"example_index": example_index,
                                        "result_file": path.relative_to(output_dir).as_posix(),
                                        "error": result["error"]})
            print(f"[{example_index}] FAILED: {result['error']}", file=sys.stderr, flush=True)
        summary["processed"] += 1
        summary["remaining"] -= 1
        save_json(output_dir / "summary.json", summary)
    print(f"Retrieval: {summary['successful']}/{len(indices)} successful; "
          f"failures={summary['failed']} reused={summary['reused']}\nOutputs: {output_dir}", flush=True)
    return output_dir / "summary.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["test", "train", "dev"], default="test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="All examples from --start")
    parser.add_argument("--max-chunks", type=int, default=30)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=.75)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Parent of content-addressed run folders")
    parser.add_argument("--prepared-cache-dir", type=Path, default=preparation.DEFAULT_CACHE)
    parser.add_argument("--index-cache-dir", type=Path, default=bm25_index.DEFAULT_CACHE)
    parser.add_argument("--repository-cache-dir", type=Path, help="Read existing clones here; never clone/fetch")
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--dataset-file", type=Path, help="Optional existing Arrow file for the selected split (offline)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate", action="store_true", help="Read labels only after all predictions have been saved")
    args = parser.parse_args(argv)
    settings = Settings(args.max_chunks, args.window, args.stride, args.k1, args.b)
    try:
        settings.validate()
        if args.start < 0 or args.limit <= 0:
            raise ValueError("Require start >= 0 and limit > 0")
    except ValueError as error:
        parser.error(str(error))
    try:
        dataset = load_dataset_input(args.split, args.dataset_cache_dir, args.dataset_file)
        if args.start >= len(dataset):
            raise ValueError("Start index is outside the selected dataset")
        stop = len(dataset) if args.all else min(len(dataset), args.start + args.limit)
        summary_path = run_examples(
            dataset, args.output_dir, range(args.start, stop), settings, args.split,
            args.prepared_cache_dir, args.index_cache_dir, args.repository_cache_dir, args.resume)
        summary = json.loads(summary_path.read_bytes())
        if args.evaluate:
            from bug_localization.rag.evaluate_retrieval import evaluate_run, print_metrics
            metrics = evaluate_run(summary_path, dataset)
            print_metrics(metrics)
            print(f"Evaluation outputs: {summary_path.parent / 'evaluation'}")
            if metrics["evaluation_failures"]:
                return 1
        return int(bool(summary["failed"]))
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
