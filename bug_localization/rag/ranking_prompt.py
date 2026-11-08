"""Versioned evidence prompt and deterministic whole-chunk budget packing."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from bug_localization.rag.retrieval import digest


PROMPT_VERSION = "file-ranking/v1"
TEMPLATE_PATH = Path(__file__).parent / "prompts" / "file_ranking_v1.txt"


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class TokenCounter:
    """Model-specific tiktoken text counts plus conservative envelope allowances.

    No tokenizer: one token per UTF-8 byte (an upper bound for byte BPE text).
    Both modes add 16 tokens/message, 32/request and the response-format JSON.
    The caller also reserves >=5% of context for undocumented server framing.
    """
    def __init__(self, model_name, mode="auto"):
        self.encoding = None
        self.identity = {"policy": "text+16/message+32/request+format-json/v1", "mode": "utf8-bytes"}
        if mode == "auto":
            try:
                import tiktoken
                self.encoding = tiktoken.encoding_for_model(model_name)
                self.identity.update(mode="tiktoken", encoding=self.encoding.name,
                                     package_version=version("tiktoken"))
            except (ImportError, KeyError, OSError, ValueError):
                self.identity["fallback_reason"] = "model_tokenizer_unavailable"

    def count_text(self, text):
        if self.encoding is None:
            return len(text.encode("utf-8"))
        return len(self.encoding.encode(text, disallowed_special=()))

    def count_request(self, messages, settings):
        return (32 + self.count_text(json_text(settings.format_parameter()))
                + sum(16 + self.count_text(m["role"]) + self.count_text(m["content"]) for m in messages))


def prompt_identity():
    return {"version": PROMPT_VERSION, "template_sha256": hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest(),
            "packing_policy": "bm25-whole-chunks-skip-oversize;issue-character-prefix/v1"}


def build_messages(issue, chunks, candidates):
    evidence = {"issue_title": issue["issue_title"], "issue_body": issue["issue_body"],
                "allowed_candidate_paths": candidates,
                "retrieved_code": [{key: chunk[key] for key in (
                    "chunk_id", "path", "start_line", "end_line", "text")} for chunk in chunks]}
    return [{"role": "system", "content": TEMPLATE_PATH.read_text(encoding="utf-8")},
            {"role": "user", "content": json_text(evidence)}]


def pack_prompt(prediction, settings, counter):
    settings.validate()
    example = prediction["identity"]["example_input"]
    original = {key: example.get(key) or "" for key in ("issue_title", "issue_body")}
    issue = dict(original)
    budget = settings.input_budget

    def count(chunks=(), candidates=()):
        return counter.count_request(build_messages(issue, chunks, list(candidates)), settings)

    if count() > budget:
        # Remove suffixes in blocks of 1/8 the remaining characters (at least 1),
        # body first, then title. Recount after every removal; no BPE monotonicity
        # assumption and no changes to source chunks. A prefix is always kept.
        for key in ("issue_body", "issue_title"):
            while issue[key] and count() > budget:
                issue[key] = issue[key][:-max(1, (len(issue[key]) + 7) // 8)]
    truncation = {key: {"original_characters": len(original[key]), "kept_characters": len(issue[key])}
                  for key in original}
    truncated = original != issue
    included, decisions = [], []
    bm25_paths = [row["file"] for row in prediction["ranking"]]

    def allowed(chunks):
        paths = {c["path"] for c in chunks}
        return [path for path in bm25_paths if path in paths]

    overhead_fits = count() <= budget
    for chunk in sorted(prediction["retrieved_chunks"], key=lambda c: (-c["score"], c["chunk_id"])):
        trial = included + [chunk]
        fits = overhead_fits and count(trial, allowed(trial)) <= budget
        decisions.append({"chunk_id": chunk["chunk_id"], "path": chunk["path"],
                          "included": fits, "reason": "included" if fits else "input_budget"})
        if fits:
            included.append(chunk)
    candidates = allowed(included)
    messages = build_messages(issue, included, candidates)
    final_count = counter.count_request(messages, settings)
    if included and final_count > budget:
        raise ValueError("Final prompt exceeds input budget")
    reason = (None if included else "empty_retrieval" if not prediction["retrieved_chunks"]
              else "prompt_overhead_exceeds_budget" if not overhead_fits else "no_evidence_fits")
    return {"prompt": prompt_identity(), "messages": messages, "allowed_candidate_paths": candidates,
            "included_chunk_ids": [c["chunk_id"] for c in included], "chunk_decisions": decisions,
            "issue_truncated": truncated, "issue_truncation": truncation,
            "token_count": final_count, "token_counter": counter.identity,
            "input_budget": budget, "reserved_output_tokens": settings.max_output_tokens,
            "reserved_safety_tokens": settings.reserved_safety_tokens,
            "fits_budget": final_count <= budget, "skip_reason": reason,
            "messages_sha256": digest(messages)}
