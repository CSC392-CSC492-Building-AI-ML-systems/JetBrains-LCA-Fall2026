"""Prepare Python bug-localization snapshots; no retrieval or label access.

Run from the repository root:
    python bug_localization/baseline_eval/prepare_repositories.py --limit 1
"""

import argparse
import hashlib
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

if __package__:
    from .run_all_tfidf import DATASET, ROOT, get_repository, read_content
else:
    from run_all_tfidf import DATASET, ROOT, get_repository, read_content


FORMAT_VERSION = 1
# Bump the format/policy version if the baseline reader's behavior changes.
CANDIDATE_POLICY = {
    "reader": "baseline_eval.run_all_tfidf.read_content/v1",
    "extension": ".py",
    "excluded_substrings_case_insensitive": ["test/", "tests/"],
    "object_type": "blob",
    "decoding": "utf-8-strict",
    "empty_files": "retain",
}
DEFAULT_CACHE = ROOT / "outputs" / "prepared_snapshots"


def validate_settings(window, stride):
    if type(window) is not int or type(stride) is not int:
        raise ValueError("window and stride must be integers")
    if window <= 0 or not 0 < stride <= window:
        raise ValueError("Require window > 0 and 0 < stride <= window")


def source_lines(text):
    """Physical source lines, preserving CR/LF endings and the final newline.

    Unlike str.splitlines(), form feeds and Unicode separators inside source
    strings are not treated as new source lines. A final newline adds no line.
    """
    return re.findall(r"[^\r\n]*(?:\r\n|\r|\n|$)", text)[:-1]


def normalize_path(path):
    path = path.replace("\\", "/")
    relative = PurePosixPath(path)
    if (not path or relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() != path or re.match(r"^[A-Za-z]:", path)):
        raise ValueError(f"Expected a repository-relative path: {path!r}")
    return path


def normalize_files(files):
    normalized = {}
    for path, text in files.items():
        path = normalize_path(path)
        if path in normalized:
            raise ValueError(f"Paths collide after normalization: {path}")
        if not isinstance(text, str):
            raise ValueError(f"Expected decoded source text: {path}")
        normalized[path] = text
    return dict(sorted(normalized.items()))


def chunk_files(files, window=40, stride=20):
    """Return deterministic windows with 1-based inclusive line numbers.

    Empty candidates stay in the file mapping but contribute no chunks.
    Chunk text is an exact slice of source, including original line endings.
    Identical content in different files is deliberately retained.
    """
    validate_settings(window, stride)
    chunks = []
    for path, text in normalize_files(files).items():
        lines = source_lines(text)
        start = 0
        while start < len(lines):
            end = min(start + window, len(lines))
            text = "".join(lines[start:end])
            if text.strip():
                chunks.append({
                    "chunk_id": len(chunks), "path": path,
                    "start_line": start + 1, "end_line": end, "text": text,
                })
            if end == len(lines):
                break
            start += stride
    return chunks


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


@dataclass
class PreparedSnapshot:
    metadata: dict
    chunks: list[dict]
    cache_path: Path
    reused: bool

    @property
    def candidate_paths(self):
        return self.metadata["candidate_paths"]


def cache_identity(repository, resolved_sha, window, stride):
    validate_settings(window, stride)
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", resolved_sha):
        raise ValueError("Cache requires the full resolved commit SHA")
    return {
        "format_version": FORMAT_VERSION,
        "repository": repository,
        "base_sha": resolved_sha,
        "candidate_policy": CANDIDATE_POLICY,
        "chunking": {
            "window": window, "stride": stride,
            "line_numbers": "1-based-inclusive",
            "line_endings": "preserve-cr-lf",
            "whitespace_chunks": "skip",
            "duplicate_chunks": "retain",
        },
    }


def load_cache(cache_path, identity):
    """Return only a complete, internally consistent cache; otherwise rebuild."""
    try:
        metadata = json.loads((cache_path / "metadata.json").read_bytes())
        expected_keys = set(identity) | {
            "candidate_paths", "file_count", "chunk_count", "chunks_sha256",
        }
        if set(metadata) != expected_keys:
            return None
        if any(metadata.get(key) != value for key, value in identity.items()):
            return None
        paths = metadata["candidate_paths"]
        if (not isinstance(paths, list) or not all(isinstance(p, str) for p in paths)
                or paths != sorted(set(paths))
                or any(normalize_path(p) != p for p in paths)
                or type(metadata["file_count"]) is not int
                or metadata["file_count"] != len(paths)):
            return None
        data = (cache_path / "chunks.jsonl").read_bytes()
        if hashlib.sha256(data).hexdigest() != metadata["chunks_sha256"]:
            return None
        chunks = [json.loads(line) for line in data.splitlines()]
        if (type(metadata["chunk_count"]) is not int
                or len(chunks) != metadata["chunk_count"]):
            return None
        candidates = set(paths)
        previous = None
        for chunk_id, chunk in enumerate(chunks):
            if set(chunk) != {"chunk_id", "path", "start_line", "end_line", "text"}:
                return None
            if (type(chunk["chunk_id"]) is not int or chunk["chunk_id"] != chunk_id
                    or chunk["path"] not in candidates
                    or type(chunk["start_line"]) is not int
                    or type(chunk["end_line"]) is not int
                    or not isinstance(chunk["text"], str) or not chunk["text"].strip()):
                return None
            start, end = chunk["start_line"], chunk["end_line"]
            order = (chunk["path"], start)
            if (start < 1 or not start <= end < start + identity["chunking"]["window"]
                    or (start - 1) % identity["chunking"]["stride"]
                    or len(source_lines(chunk["text"])) != end - start + 1
                    or (previous is not None and order <= previous)):
                return None
            previous = order
        return PreparedSnapshot(metadata, chunks, cache_path, reused=True)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def prepare_snapshot(repository, repository_identity, base_sha,
                     cache_root=DEFAULT_CACHE, window=40, stride=20):
    """Prepare from Git code and provenance only; no issue, head SHA, or diff.

    Resolve before checking the cache so abbreviated inputs cannot produce
    ambiguous identities. read_content is the unchanged TF-IDF candidate reader.
    """
    validate_settings(window, stride)
    resolved_sha = repository.commit(base_sha).hexsha
    identity = cache_identity(repository_identity, resolved_sha, window, stride)
    key = hashlib.sha256(json_bytes(identity)).hexdigest()
    cache_path = Path(cache_root) / key
    cached = load_cache(cache_path, identity)
    if cached is not None:
        return cached

    files = normalize_files(read_content(repository, resolved_sha))
    chunks = chunk_files(files, window, stride)
    data = b"".join(json_bytes(chunk) for chunk in chunks)
    metadata = {
        **identity, "candidate_paths": list(files), "file_count": len(files),
        "chunk_count": len(chunks), "chunks_sha256": hashlib.sha256(data).hexdigest(),
    }
    cache_path.mkdir(parents=True, exist_ok=True)
    # Unique temporary files also avoid collisions between concurrent writers.
    # Metadata is the completion marker and is replaced only after chunks exist.
    # The checksum rejects an interrupted or mismatched pair of files.
    with tempfile.TemporaryDirectory(prefix=".writing-", dir=cache_path) as work:
        temporary = Path(work)
        (temporary / "chunks.jsonl").write_bytes(data)
        (temporary / "metadata.json").write_bytes(json_bytes(metadata))
        (temporary / "chunks.jsonl").replace(cache_path / "chunks.jsonl")
        (temporary / "metadata.json").replace(cache_path / "metadata.json")
    return PreparedSnapshot(metadata, chunks, cache_path, reused=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # The reusable TF-IDF reader is Python-only; do not imply parity for other languages.
    parser.add_argument("--split", choices=["dev", "train", "test"], default="test")
    parser.add_argument("--start", type=int, default=0, help="First example index (0-based)")
    parser.add_argument("--limit", type=int, default=1, help="Number of examples (default: smoke run)")
    parser.add_argument("--all", action="store_true", help="Prepare every example from --start")
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset-cache-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_settings(args.window, args.stride)
        if args.start < 0 or args.limit <= 0:
            raise ValueError("Require start >= 0 and limit > 0")
    except ValueError as error:
        parser.error(str(error))

    try:
        from datasets import load_dataset

        dataset = load_dataset(DATASET, "py", split=args.split,
                               cache_dir=str(args.dataset_cache_dir) if args.dataset_cache_dir else None)
        if args.start >= len(dataset):
            raise ValueError(f"Start index {args.start} is outside {len(dataset)} examples")
    except Exception as error:
        print(f"FAILED loading dataset: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    stop = len(dataset) if args.all else min(len(dataset), args.start + args.limit)
    failures = 0
    for index in range(args.start, stop):
        repository = None
        name = f"example {index}"
        try:
            # Project to provenance before repository preparation. No labels or
            # issue text enter the preparation API, records, or cache identity.
            row = dataset[index]
            example = {key: row[key] for key in ("repo_owner", "repo_name", "base_sha")}
            name = f"{example['repo_owner']}__{example['repo_name']}"
            repository = get_repository(example)
            prepared = prepare_snapshot(repository, name, example["base_sha"],
                                        args.cache_dir, args.window, args.stride)
            metadata = prepared.metadata
            print(f"[{index}] {name} {metadata['base_sha']} "
                  f"files={metadata['file_count']} chunks={metadata['chunk_count']} "
                  f"cache={'reused' if prepared.reused else 'written'} {prepared.cache_path}", flush=True)
        except Exception as error:
            failures += 1
            print(f"[{index}] FAILED {name}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        finally:
            if repository is not None:
                repository.close()
    print(f"Prepared {stop - args.start - failures}/{stop - args.start} examples; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
