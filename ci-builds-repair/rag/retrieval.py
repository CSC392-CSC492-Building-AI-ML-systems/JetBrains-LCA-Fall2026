"""Deterministic snapshot chunking and label-free BM25 retrieval.

Adapted from Talha's rag-bug-localization branch (23129da). CI repair retains
tests and text configuration files because they can be repair targets.
"""

import keyword
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project_level_code_completion.rag.bm25 import BM25Index, CAMEL_BOUNDARY, IDENTIFIER

EXTENSIONS = {".py", ".pyi", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".txt", ".md", ".rst", ".json"}
KEYWORDS = frozenset(keyword.kwlist)


def source_lines(text):
    return re.findall(r"[^\r\n]*(?:\r\n|\r|\n|$)", text)[:-1]


def tokenize(text):
    terms = []
    for identifier in IDENTIFIER.findall(text):
        whole = identifier.lower()
        parts = CAMEL_BOUNDARY.sub("_", identifier).split("_")
        terms.extend(term for term in [whole, *(part.lower() for part in parts if part.lower() != whole)]
                     if len(term) >= 2 and term not in KEYWORDS)
    return terms


def chunk_files(files, window=40, stride=20):
    if window < 1 or not 0 < stride <= window:
        raise ValueError("Require window > 0 and 0 < stride <= window")
    chunks = []
    for path, text in sorted(files.items()):
        lines = source_lines(text)
        for start in range(0, len(lines), stride):
            end = min(start + window, len(lines))
            content = "".join(lines[start:end])
            if content.strip():
                chunks.append({"chunk_id": len(chunks), "path": path, "start": start + 1,
                               "end": end, "text": content})
            if end == len(lines):
                break
    return chunks


def make_query(logs, workflow_name=""):
    # This function never receives diff, changed_files, or successful revision.
    lines = []
    for log in logs:
        text = log.get("log", "") if isinstance(log, dict) else str(log)
        lines.extend(text.splitlines())
    first = next((i for i, line in enumerate(lines) if "error" in line.lower()), None)
    excerpt = lines[-7:] if first is None else lines[max(0, first - 3):first + 4]
    return "\n".join([workflow_name, *excerpt])


def retrieve(chunks, query, limit=30):
    index = BM25Index([tokenize(chunk["path"] + "\n" + chunk["text"]) for chunk in chunks], k1=1.2, b=0.75)
    scores = defaultdict(float)
    # The core scorer iterates a set; single sorted terms make sums reproducible.
    for term in sorted(set(tokenize(query))):
        for chunk_id, value in index.scores([term]).items():
            scores[chunk_id] += value
    ranked = sorted(scores, key=lambda i: (-scores[i], i))
    return [{**chunks[i], "score": scores[i]} for i in ranked[:limit] if scores[i] > 0]
