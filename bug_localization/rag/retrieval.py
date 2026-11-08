"""Label-free bug-report queries, chunk evidence, and max-chunk file rankings."""

import hashlib
import math

from bug_localization.rag import prepare_repositories as preparation


PREDICTION_FIELDS = (
    "repo_owner", "repo_name", "base_sha", "issue_url", "issue_title", "issue_body",
)
RETRIEVAL_VERSION = 1


def digest(value):
    return hashlib.sha256(preparation.json_bytes(value)).hexdigest()


def project_example(row):
    """Read only prediction inputs; never inspect labels, head SHA, or diffs."""
    example = {key: row[key] for key in ("repo_owner", "repo_name", "base_sha")}
    for key in ("issue_url", "issue_title", "issue_body"):
        value = row.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be text or null")
        example[key] = value or ""
    return example


def make_query(example):
    """Join nonblank issue_title and issue_body with a blank line.

    Preserve the supplied content, including indentation, CR/LF, Markdown,
    stack traces, and snippets. Null/missing/whitespace-only fields add nothing.
    """
    parts = []
    for key in ("issue_title", "issue_body"):
        value = example.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be text or null")
        if value and value.strip():
            parts.append(value)
    return "\n\n".join(parts)


def retrieve(index, query, max_chunks=30):
    """Score every matching indexed chunk before applying the evidence limit."""
    if type(max_chunks) is not int or max_chunks < 0:
        raise ValueError("max_chunks must be a nonnegative integer")
    matches = index.search(query, max_results=index.bm25.n_docs)
    file_scores = dict.fromkeys(index.snapshot.candidate_paths, 0.0)
    for chunk in matches:
        file_scores[chunk["path"]] = max(file_scores[chunk["path"]], chunk["score"])
    ranking = [{"rank": rank, "file": path, "score": score}
               for rank, (path, score) in enumerate(
                   sorted(file_scores.items(), key=lambda item: (-item[1], item[0])), 1)]
    chunks = [{"rank": rank, **chunk} for rank, chunk in enumerate(matches[:max_chunks], 1)]
    return {
        "retrieved_chunks": chunks, "ranking": ranking,
        "candidate_count": len(file_scores),
        "distinct_retrieved_files": len({chunk["path"] for chunk in chunks}),
    }


def validate_prediction(prediction, index, max_chunks):
    """Check persisted predictions without rescoring or reading any labels."""
    ranking = prediction["ranking"]
    paths = [row["file"] for row in ranking]
    if (len(paths) != len(set(paths)) or set(paths) != set(index.snapshot.candidate_paths)
            or prediction["candidate_count"] != len(paths)
            or [row["rank"] for row in ranking] != list(range(1, len(paths) + 1))
            or any(not math.isfinite(row["score"]) or row["score"] < 0 for row in ranking)
            or ranking != sorted(ranking, key=lambda row: (-row["score"], row["file"]))):
        raise ValueError("Incomplete or invalid file ranking")
    chunks = prediction["retrieved_chunks"]
    ids = [chunk["chunk_id"] for chunk in chunks]
    if (len(chunks) > max_chunks or len(set(ids)) != len(ids)
            or [c["rank"] for c in chunks] != list(range(1, len(chunks) + 1))
            or chunks != sorted(chunks, key=lambda c: (-c["score"], c["chunk_id"]))
            or prediction["distinct_retrieved_files"] != len({c["path"] for c in chunks})):
        raise ValueError("Invalid retrieved chunk order/count")
    scores = {row["file"]: row["score"] for row in ranking}
    for chunk in chunks:
        original = index.chunks_by_id[chunk["chunk_id"]]
        if (chunk["chunk_id"] not in index.chunk_ids
                or not math.isfinite(chunk["score"]) or chunk["score"] <= 0
                or any(chunk[key] != value for key, value in original.items())
                or scores[chunk["path"]] < chunk["score"]):
            raise ValueError("Retrieved evidence does not match the prepared snapshot")
