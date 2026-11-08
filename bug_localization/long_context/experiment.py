"""Label-free snapshot preparation and shared prediction/validation for every mode."""

import time
from dataclasses import asdict
from pathlib import Path

from bug_localization.baseline_eval.run_all_tfidf import get_repository, read_content
from bug_localization.rag.bm25_index import ChunkBM25Index
from bug_localization.rag.file_ranking import complete_ranking, validate_response
from bug_localization.rag.model import ModelResponse, OpenAIClient
from bug_localization.rag.prepare_repositories import PreparedSnapshot, chunk_files, normalize_files
from bug_localization.rag.retrieval import digest, make_query, project_example, retrieve
from bug_localization.long_context.prompt import pack_prompt


MODEL_STATUSES = {"success", "repaired"}
FALLBACK_STATUSES = {"fallback_request", "fallback_validation", "fallback_context"}
COMPLETE_STATUSES = MODEL_STATUSES | FALLBACK_STATUSES
TAIL_POLICY = "BM25 max-chunk file score descending; ties lexicographic; all unreturned candidates appended"


def load_snapshot(example):
    """Use the baseline Git helpers, never the worktree or the example's head SHA."""
    provenance = {key: example[key] for key in ("repo_owner", "repo_name", "base_sha")}
    repository = get_repository(provenance)
    try:
        resolved_sha = repository.commit(provenance["base_sha"]).hexsha
        files = normalize_files(read_content(repository, resolved_sha))
    finally:
        repository.close()
    if not files:
        raise ValueError("No candidate files")
    return resolved_sha, files


def prepare_input(example, window=40, stride=20):
    example = project_example(example)
    resolved_sha, files = load_snapshot(example)
    chunks = chunk_files(files, window, stride)
    metadata = {"candidate_paths": list(files)}
    snapshot = PreparedSnapshot(metadata, chunks, Path("."), False)
    index = ChunkBM25Index(snapshot)  # Same default k1=1.2, b=.75 as existing RAG.
    retrieval = retrieve(index, make_query(example), max_chunks=len(chunks))
    return {"example_input": example, "resolved_base_sha": resolved_sha, "files": files,
            "snapshot_sha256": digest(files), "retrieval": retrieval}


def predict(prepared, mode, repo_token_budget, settings, counter, client=None, dry_run=False):
    packed = pack_prompt(prepared["example_input"], prepared["files"],
                         prepared["retrieval"]["retrieved_chunks"], mode, repo_token_budget, settings, counter)
    baseline = prepared["retrieval"]["ranking"]
    result = {"packing": packed, "bm25_ranking": baseline, "ranking": baseline,
              "tail_policy": TAIL_POLICY, "validation": None, "model_response": None,
              "model_ranking_success": False, "bm25_fallback": False,
              "model_ranked_paths": [], "submitted_messages": None,
              "request_latency_seconds": 0.0}
    if dry_run:
        return {**result, "status": "dry_run"}
    if not packed["fits_context"]:
        return {**result, "status": "fallback_context", "bm25_fallback": True,
                "fallback_reason": packed["skip_reason"]}
    result["submitted_messages"] = packed["messages"]
    started = time.perf_counter()
    try:
        response = (client or OpenAIClient(settings)).generate(packed["messages"])
    except Exception as error:
        response = ModelResponse(None, error={"type": type(error).__name__, "stage": "model_client"},
                                 latency_seconds=time.perf_counter() - started)
    result["model_response"] = asdict(response)
    result["request_latency_seconds"] = response.latency_seconds
    if response.error:
        result.update(status="fallback_request", bm25_fallback=True, fallback_reason="request_failure")
    elif response.refusal or response.finish_reason not in (None, "stop"):
        result.update(status="fallback_validation", bm25_fallback=True,
                      fallback_reason="refusal_or_incomplete_response")
    else:
        # Input candidates are lexicographic. Only the shared deterministic tail
        # uses BM25 order, including in issue_only. The model sees no BM25 scores.
        allowed = [row["file"] for row in baseline]
        validation = validate_response(response.raw_response, allowed)
        result.update(status=validation["status"], validation=validation)
        if validation["status"] in MODEL_STATUSES:
            omitted = next((len(p["paths"]) for p in validation["problems"]
                            if p["type"] == "omitted_candidates"), 0)
            result["model_ranked_paths"] = validation["ranking"][:len(validation["ranking"]) - omitted]
            result["ranking"] = complete_ranking(validation["ranking"], baseline)
            result["model_ranking_success"] = True
        else:
            result.update(bm25_fallback=True, fallback_reason="unusable_response")
    return result


def validate_saved(result, manifest, run_signature):
    """Reject changed provenance, invalid permutations and mismatched model outcomes."""
    index = result["example_index"]
    if (result["run_signature"] != run_signature or index not in manifest["example_indices"]
            or digest(result["example_input"]) != manifest["input_sha256"][str(index)]
            or result["mode"] != manifest["mode"]
            or result["repo_token_budget"] != manifest["repo_token_budget"]):
        raise ValueError("Prediction identity mismatch")
    baseline = result["bm25_ranking"]
    paths = [r["file"] for r in baseline]
    packing = result["packing"]
    if (len(paths) != len(set(paths)) or sorted(paths) != packing["allowed_candidate_paths"]
            or paths != [r["file"] for r in sorted(baseline, key=lambda r: (-r["score"], r["file"]))]
            or result["candidate_count"] != len(paths)
            or digest(packing["messages"]) != packing["messages_sha256"]
            or result["tail_policy"] != TAIL_POLICY):
        raise ValueError("Candidate, prompt or tail mismatch")
    status = result["status"]
    if status not in COMPLETE_STATUSES | {"dry_run"}:
        raise ValueError("Incomplete prediction")
    if result["model_ranking_success"] != (status in MODEL_STATUSES):
        raise ValueError("Model success flag mismatch")
    if result["bm25_fallback"] != (status in FALLBACK_STATUSES):
        raise ValueError("Fallback flag mismatch")
    if status in MODEL_STATUSES:
        validation = validate_response(result["model_response"]["raw_response"], paths)
        if validation != result["validation"] or validation["status"] != status:
            raise ValueError("Model response validation mismatch")
        expected = complete_ranking(validation["ranking"], baseline)
    else:
        expected = baseline
    if result["ranking"] != expected:
        raise ValueError("Invalid final ranking")
    if result["submitted_messages"] is not None:
        if (result["submitted_messages"] != packing["messages"] or not packing["fits_context"]
                or packing["actual_source_tokens"] > manifest["repo_token_budget"]
                or packing["total_reserved_tokens"] > manifest["model"]["context_window"]):
            raise ValueError("Oversized or changed submitted request")
