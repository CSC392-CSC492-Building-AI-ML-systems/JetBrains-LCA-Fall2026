"""Build/reload a BM25 chunk index for ONE existing step-1 prepared snapshot.

No dataset loading, file aggregation, evaluation, or generation is performed.
Run from the repository root, or invoke this file directly from any directory.
"""

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bug_localization.rag import prepare_repositories as preparation
from bug_localization.rag import retrieval_tokenizer as tokenizer
from project_level_code_completion.rag.bm25 import BM25Index


FORMAT_VERSION = 1
# Bump when the reused core or our scoring/persistence semantics change.
SCORING_VERSION = "adriana-okapi-nonnegative-idf/distinct-query-sorted-sum/v1"
DEFAULT_CACHE = preparation.RAG_ROOT / "outputs" / "bm25_indexes"


def validate_settings(k1, b):
    if (isinstance(k1, bool) or not isinstance(k1, (int, float))
            or not math.isfinite(k1) or k1 <= 0):
        raise ValueError("k1 must be finite and greater than zero")
    if (isinstance(b, bool) or not isinstance(b, (int, float))
            or not math.isfinite(b) or not 0 <= b <= 1):
        raise ValueError("b must be finite and between zero and one")


def load_prepared_snapshot(cache_path):
    """Read step 1's validated cache without loading Git or the dataset.

    Reject stale/incomplete preparation rather than repairing source evidence.
    Regenerate it with the preparation CLI if this raises ValueError.
    """
    cache_path = Path(cache_path)
    try:
        metadata = json.loads((cache_path / "metadata.json").read_bytes())
        identity = preparation.cache_identity(
            metadata["repository"], metadata["base_sha"],
            metadata["chunking"]["window"], metadata["chunking"]["stride"])
        snapshot = preparation.load_cache(cache_path, identity)
        if snapshot is not None:
            return snapshot
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    raise ValueError(f"Invalid or incomplete prepared snapshot: {cache_path}; rerun preparation")


def cache_identity(snapshot, k1, b):
    # Hash actual records as well as provenance: in-memory edits cannot silently
    # reuse an index merely because a caller retained old preparation metadata.
    chunks_hash = hashlib.sha256()
    for chunk in snapshot.chunks:
        chunks_hash.update(preparation.json_bytes(chunk))
    return {
        "format_version": FORMAT_VERSION,
        "prepared_metadata_sha256": hashlib.sha256(
            preparation.json_bytes(snapshot.metadata)).hexdigest(),
        "prepared_chunks_sha256": chunks_hash.hexdigest(),
        "tokenizer": tokenizer.tokenizer_identity(),
        "scoring_version": SCORING_VERSION,
        "k1": float(k1), "b": float(b),
    }


class ChunkBM25Index:
    """Snapshot evidence plus Adriana's small, dependency-free inverted index.

    ``bm25`` exposes n_docs, doc_len, avg_len, postings (doc index, term
    frequency), and idf(). ``chunk_ids`` maps its dense document indexes back
    to original IDs, while ``chunks_by_id`` includes excluded evidence too.
    """

    def __init__(self, snapshot, k1=1.2, b=0.75):
        validate_settings(k1, b)
        self.snapshot = snapshot
        self.chunks_by_id = {chunk["chunk_id"]: chunk for chunk in snapshot.chunks}
        if len(self.chunks_by_id) != len(snapshot.chunks):
            raise ValueError("Prepared snapshot has duplicate chunk IDs")
        documents, self.chunk_ids = [], []
        for chunk in snapshot.chunks:
            terms = tokenizer.tokenize(chunk["text"])
            if terms:
                self.chunk_ids.append(chunk["chunk_id"])
                documents.append(terms)
        self.bm25 = BM25Index(documents, k1=k1, b=b)
        self.cache_path = None
        self.reused = False

    @property
    def excluded_count(self):
        return len(self.snapshot.chunks) - self.bm25.n_docs

    @property
    def document_frequency(self):
        return {term: len(entries) for term, entries in self.bm25.postings.items()}

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        """Positive matches, descending by score; exact ties use original ID.

        Use the unchanged core scorer once per distinct term, in sorted order,
        to make floating-point accumulation independent of Python's hash seed.
        Results include the original text and 1-based inclusive source ranges.
        """
        if type(max_results) is not int or max_results < 0:
            raise ValueError("max_results must be a nonnegative integer")
        if max_results == 0:
            return []
        scores = defaultdict(float)
        for term in sorted(set(tokenizer.tokenize(query))):
            for doc_id, score in self.bm25.scores([term]).items():
                scores[doc_id] += score
        ranked = sorted(((doc_id, score) for doc_id, score in scores.items() if score > 0),
                        key=lambda item: (-item[1], self.chunk_ids[item[0]]))
        return [{**self.chunks_by_id[self.chunk_ids[doc_id]], "score": score}
                for doc_id, score in ranked[:max_results]]


def _load_index(cache_path, identity, snapshot):
    """Validate checksum, completion marker, mappings, and posting invariants."""
    try:
        metadata = json.loads((cache_path / "metadata.json").read_bytes())
        stats = {"index_sha256", "indexed_count", "excluded_count", "vocabulary_size", "avg_len"}
        if (set(metadata) != set(identity) | stats
                or any(metadata[key] != value for key, value in identity.items())):
            return None
        data = (cache_path / "index.json").read_bytes()
        if hashlib.sha256(data).hexdigest() != metadata["index_sha256"]:
            return None
        state = json.loads(data)
        if set(state) != {"chunk_ids", "doc_len", "postings"}:
            return None
        ids, lengths, postings = state["chunk_ids"], state["doc_len"], state["postings"]
        if (not isinstance(ids, list) or not isinstance(lengths, list)
                or not isinstance(postings, dict)
                or any(type(i) is not int for i in ids)
                or len(ids) != len(lengths) or ids != sorted(set(ids))
                or not set(ids).issubset(c["chunk_id"] for c in snapshot.chunks)
                or any(type(n) is not int or n <= 0 for n in lengths)):
            return None
        expected_stats = {
            "indexed_count": len(ids), "excluded_count": len(snapshot.chunks) - len(ids),
            "vocabulary_size": len(postings), "avg_len": sum(lengths) / len(ids) if ids else 0.0,
        }
        if (any(metadata[key] != value for key, value in expected_stats.items())
                or any(type(metadata[key]) is not int
                       for key in ("indexed_count", "excluded_count", "vocabulary_size"))):
            return None
        totals = [0] * len(ids)
        for term, entries in postings.items():
            if not isinstance(term, str) or not term or not isinstance(entries, list) or not entries:
                return None
            previous = -1
            for entry in entries:
                if not isinstance(entry, list) or len(entry) != 2:
                    return None
                doc_id, tf = entry
                if (type(doc_id) is not int or type(tf) is not int
                        or not previous < doc_id < len(ids) or tf <= 0):
                    return None
                totals[doc_id] += tf
                previous = doc_id
        if totals != lengths:
            return None
        # Restore sufficient statistics directly: no source re-tokenization.
        index = ChunkBM25Index.__new__(ChunkBM25Index)
        index.snapshot = snapshot
        index.chunks_by_id = {chunk["chunk_id"]: chunk for chunk in snapshot.chunks}
        index.chunk_ids = ids
        index.bm25 = BM25Index([], k1=identity["k1"], b=identity["b"])
        index.bm25.n_docs = len(ids)
        index.bm25.doc_len = lengths
        index.bm25.avg_len = expected_stats["avg_len"]
        index.bm25.postings = defaultdict(list, {
            term: [tuple(entry) for entry in entries] for term, entries in postings.items()})
        index.cache_path, index.reused = cache_path, True
        return index
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def build_or_load_index(snapshot, cache_root=DEFAULT_CACHE, k1=1.2, b=0.75):
    """Return one content-addressed index for a prepared snapshot and settings."""
    validate_settings(k1, b)
    identity = cache_identity(snapshot, k1, b)
    key = hashlib.sha256(preparation.json_bytes(identity)).hexdigest()
    cache_path = Path(cache_root) / key
    cached = _load_index(cache_path, identity, snapshot)
    if cached is not None:
        return cached
    index = ChunkBM25Index(snapshot, k1, b)
    data = preparation.json_bytes({
        "chunk_ids": index.chunk_ids, "doc_len": index.bm25.doc_len,
        "postings": dict(index.bm25.postings),
    })
    metadata = {
        **identity, "index_sha256": hashlib.sha256(data).hexdigest(),
        "indexed_count": index.bm25.n_docs, "excluded_count": index.excluded_count,
        "vocabulary_size": len(index.bm25.postings), "avg_len": index.bm25.avg_len,
    }
    cache_path.mkdir(parents=True, exist_ok=True)
    # Follow step 1: unique temporary writes, atomic replace, metadata last.
    with tempfile.TemporaryDirectory(prefix=".writing-", dir=cache_path) as work:
        temporary = Path(work)
        (temporary / "index.json").write_bytes(data)
        (temporary / "metadata.json").write_bytes(preparation.json_bytes(metadata))
        (temporary / "index.json").replace(cache_path / "index.json")
        (temporary / "metadata.json").replace(cache_path / "metadata.json")
    index.cache_path = cache_path
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_snapshot", type=Path, help="Step-1 directory containing metadata.json and chunks.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--query", help="Optional diagnostic query text")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        validate_settings(args.k1, args.b)
        if args.top_k < 0:
            raise ValueError("top-k must be nonnegative")
    except ValueError as error:
        parser.error(str(error))
    try:
        snapshot = load_prepared_snapshot(args.prepared_snapshot)
        index = build_or_load_index(snapshot, args.cache_dir, args.k1, args.b)
        print(f"repository={snapshot.metadata['repository']} base_sha={snapshot.metadata['base_sha']}")
        print(f"indexed={index.bm25.n_docs} excluded={index.excluded_count} "
              f"vocabulary={len(index.bm25.postings)} avg_len={index.bm25.avg_len:.6f} "
              f"k1={index.bm25.k1} b={index.bm25.b}")
        print(f"cache={'reused' if index.reused else 'written'} {index.cache_path}")
        if args.query is not None:
            results = index.search(args.query, args.top_k)
            print(f"matches={len(results)}")
            for result in results:
                print(f"{result['score']:.8f} chunk_id={result['chunk_id']} "
                      f"{result['path']}:{result['start_line']}-{result['end_line']}")
    except (OSError, ValueError) as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
