"""Read-only repository tools for the planner (RAG_EXPERIMENT_LOG sections 8.80-8.82).

The tools see exactly what RAG sees: the chunks of `RagContextBuilder`, which never include the completion file.
Lines are numbered from 1 in everything the planner reads or writes. The planner starts from RAG's own context (the
"default") and can only edit it: add repository code, drop default chunks, or change nothing.
"""
import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from rag.bm25 import tokenize_code
from rag.builder import RagContextBuilder, reciprocal_rank_fusion
from rag.chunker import Chunk


@dataclass(frozen=True)
class Limits:
    tool_calls: int = 5  # tool calls before the planner must submit
    search_results: int = 5
    read_lines: int = 80  # longest range read_file returns or an added range may have
    tool_output_tokens: int = 1200  # a tool result is cut to this many tokens, by whole lines
    result_chunk_tokens: int = 1200  # each chunk in a search or definition result is cut to this many tokens
    seed_chunk_tokens: int = 300  # each default chunk is shown to the planner cut to this many tokens
    definition_lines: int = 25  # lines shown for a definition found by text search
    query_characters: int = 300
    max_items: int = 8  # most items one submit_context may add
    invalid_actions: int = 2  # unusable replies tolerated before the planner is abandoned
    # "front" (section 8.82): added code goes first (nearest the prefix) and the planner may drop default chunks.
    # "append" (section 8.85): the planner can only add; added code goes last (lowest priority) and, if there is no room,
    # replaces at most `max_evict` of the lowest-ranked default chunks; at most `max_add` additions.
    edit_mode: str = "front"
    max_add: int = 2
    max_evict: int = 2
    # "swap" (section 8.93): like "append", but the planner names the (at most max_evict) default chunks to give up, and
    # only together with an addition; if the budget still does not fit, the lowest-ranked default chunks go (as in append).
    # Tools of v3 (section 8.93):
    search_mode: str = "bm25"  # "bm25", or "hybrid" = reciprocal-rank fusion of BM25 and a dense ranking (needs a dense ranker)
    dense_candidates: int = 50  # how many of each ranking enter the fusion
    usage_results: int = 5  # files shown by find_usages
    usage_context: int = 2  # lines before and after a usage
    outline_symbols: int = 80  # most definitions listed by file_outline
    dedupe_searches: bool = False  # an exact repeat of an earlier search is answered with a note, not with results
    # do-no-harm gate (8.93 amendment): an edit is used only if the planner names (in "missing") an identifier that an added
    # chunk contains and that neither the current context nor the file prefix contains; otherwise the RAG context is kept
    gate: bool = False
    imports_view: bool = False  # show the planner the repository files the code imports, with their top-level definitions


def covers(outer: Chunk, inner: Chunk) -> bool:
    return outer.path == inner.path and outer.start_line <= inner.start_line and inner.end_line <= outer.end_line


class DenseSearch:
    """Dense ranking of one repository snapshot's chunks, best first. The chunks are embedded on first use only, so a
    datapoint whose planner never searches pays nothing. `embedder` is a `rag.dense.DenseEmbedder` (or anything with
    `embed_chunks(texts)` and `embed_query(text)`); the chunk text is embedded together with its path, as in RAG."""

    def __init__(self, builder: RagContextBuilder, embedder):
        self.builder, self.embedder, self.vectors = builder, embedder, None

    def __call__(self, query: str) -> list[int]:
        if self.vectors is None:
            self.vectors = np.asarray(
                self.embedder.embed_chunks([f"{chunk.path}\n{chunk.text}" for chunk in self.builder.chunks]), dtype=np.float32)
        similarity = self.vectors @ np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        return [int(i) for i in np.argsort(-similarity, kind="stable")]


NAME = re.compile(r"[A-Za-z_]\w*")
DEFINITION_LINE = re.compile(r"^(\s*)(?:async\s+)?(def|class)\s+(\w+)")
IMPORT_LINE = re.compile(r"^\s*(?:from\s+\S+\s+import|import)\s")


class RepoTools:
    def __init__(self, builder: RagContextBuilder, limits: Limits, count_tokens: Callable[[str], int],
                 dense_ranker: Optional[Callable[[str], list[int]]] = None):
        if limits.search_mode not in ("bm25", "hybrid"):
            raise ValueError(f"search_mode must be bm25 or hybrid, got {limits.search_mode!r}")
        if limits.search_mode == "hybrid" and dense_ranker is None:
            raise ValueError("search_mode=hybrid needs a dense ranker")
        self.builder, self.limits, self.count_tokens, self.dense_ranker = builder, limits, count_tokens, dense_ranker
        self.seen: set[int] = set()  # chunk ids the planner has been shown; submit_context may cite only these
        self.asked: set[tuple[str, str]] = set()  # (tool, normalised argument) already run, for dedupe_searches
        self.chunk_id = {id(chunk): i for i, chunk in enumerate(builder.chunks)}

    def _repeat(self, tool: str, argument: str) -> bool:
        """True if dedupe is on and this exact call was made before (it is recorded either way)."""
        key = (tool, " ".join(argument.lower().split()))
        repeated = self.limits.dedupe_searches and key in self.asked
        self.asked.add(key)
        return repeated

    def _clip(self, text: str, tokens: Optional[int] = None) -> str:
        cap = tokens or self.limits.tool_output_tokens
        lines = text.split("\n")
        while len(lines) > 1 and self.count_tokens("\n".join(lines)) > cap:
            lines.pop()
        return "\n".join(lines)

    def _chunk_view(self, chunk_id: int, clip: Optional[int] = None, **extra) -> dict:
        chunk = self.builder.chunks[chunk_id]
        self.seen.add(chunk_id)
        return {"chunk": chunk_id, **extra, "path": chunk.path, "start_line": chunk.start_line + 1,
                "end_line": chunk.end_line + 1, "symbol": chunk.symbol,
                "text": self._clip(chunk.text, clip or self.limits.result_chunk_tokens)}

    def seed_view(self, default_chunks: list[Chunk]) -> list[dict]:
        """The default context as the planner sees it: best first, each chunk with its id and rank."""
        return [self._chunk_view(self.chunk_id[id(chunk)], self.limits.seed_chunk_tokens, rank=rank)
                for rank, chunk in enumerate(default_chunks, start=1)]

    def search_code(self, query) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > self.limits.query_characters:
            return {"error": f"query must be a non-empty string of at most {self.limits.query_characters} characters"}
        if self._repeat("search_code", query):
            return {"note": "you already ran exactly this search; use different words, or submit"}
        ranked = [chunk_id for chunk_id, _ in self.builder.index.rank(tokenize_code(query))]
        if self.limits.search_mode == "hybrid":
            n = self.limits.dense_candidates
            fused = reciprocal_rank_fusion([ranked[:n], self.dense_ranker(query)[:n]])
            ranked = [chunk_id for chunk_id, _ in fused]
        return {"results": [self._chunk_view(chunk_id) for chunk_id in ranked[:self.limits.search_results]]}

    def imported_view(self, prefix: str, max_files: int = 8, max_names: int = 12) -> list[dict]:
        """Repository files that the code above the line imports (as RAG resolves imports), nearest first, each with the names
        of its top-level functions and classes, so the planner need not spend tool calls to find them."""
        paths = sorted({self.builder.chunks[i].path for i in self.builder._imported_chunk_ids(prefix)},
                       key=lambda p: (-self.builder._path_proximity(p), p))[:max_files]
        view = []
        for path in paths:
            names = [m.group(3) for m in map(DEFINITION_LINE.match, self.builder.file_lines[path]) if m and not m.group(1)]
            view.append({"path": path, "defines": names[:max_names]})
        return view

    def find_usages(self, name) -> dict:
        """Where a function, class, method or variable is used (not defined or imported): one place per file, files
        nearest the completion file first, with a few lines around it. Python files only."""
        if not isinstance(name, str) or not name.strip():
            return {"error": "name must be a non-empty string"}
        name = name.strip().rstrip("()").split(".")[-1]
        if not NAME.fullmatch(name):
            return {"error": "name must be an identifier"}
        if self._repeat("find_usages", name):
            return {"note": "you already asked for the usages of this name"}
        use = re.compile(rf"\b{re.escape(name)}\b")
        context = self.limits.usage_context
        found = []
        for path in sorted(self.builder.file_lines, key=lambda p: (-self.builder._path_proximity(p), p)):
            if not path.endswith(".py"):
                continue
            lines = self.builder.file_lines[path]
            for index, line in enumerate(lines):
                definition = DEFINITION_LINE.match(line)
                if use.search(line) and not IMPORT_LINE.match(line) and not (definition and definition.group(3) == name):
                    start, end = max(0, index - context), min(len(lines), index + context + 1)
                    found.append({"path": path, "start_line": start + 1, "end_line": end,
                                  "text": self._clip("\n".join(lines[start:end]), 300),
                                  "note": "cite this with path, start_line and end_line"})
                    break
            if len(found) >= self.limits.usage_results:
                break
        return {"results": found} if found else {"error": "no usage of that name in the repository", "name": name}

    def file_outline(self, path) -> dict:
        """The functions and classes of a repository file (not the file being completed) with their line numbers."""
        lines = self.builder.file_lines.get(path) if isinstance(path, str) else None
        if lines is None:
            return {"error": "path is not a file of the repository (the file being completed is not available)"}
        entries = [f"{index + 1}: {line.strip()[:120]}" for index, line in enumerate(lines) if DEFINITION_LINE.match(line)]
        return {"path": path, "lines": len(lines), "definitions": entries[:self.limits.outline_symbols],
                "more_definitions": max(0, len(entries) - self.limits.outline_symbols)}

    def _definition_by_text(self, name: str) -> list[dict]:
        """Lines that define `name` as a function, class or assigned name, in files nearest the completion file first."""
        pattern = re.compile(rf"^\s*(?:(?:async\s+)?def|class)\s+{re.escape(name)}\b|^\s*{re.escape(name)}\s*(?::[^=\n]+)?=[^=]")
        paths = sorted(self.builder.file_lines, key=lambda p: (-self.builder._path_proximity(p), p))
        found = []
        for path in paths:
            lines = self.builder.file_lines[path]
            for index, line in enumerate(lines):
                if pattern.match(line):
                    end = min(len(lines), index + self.limits.definition_lines)
                    found.append({"path": path, "start_line": index + 1, "end_line": end,
                                  "text": self._clip("\n".join(lines[index:end])),
                                  "note": "cite this with path, start_line and end_line"})
                    break
            if len(found) >= 3:
                break
        return found

    def find_definition(self, name) -> dict:
        """Where a function, class or assigned name is defined. `obj.method` and `name()` are reduced to `method`."""
        if not isinstance(name, str) or not name.strip():
            return {"error": "name must be a non-empty string"}
        name = name.strip().rstrip("()").split(".")[-1]
        if name in self.builder.symbol_index:
            near = sorted(self.builder.symbol_index[name],
                          key=lambda i: (-self.builder._path_proximity(self.builder.chunks[i].path), i))
            return {"results": [self._chunk_view(chunk_id) for chunk_id in near[:3]]}
        found = self._definition_by_text(name)
        if found:
            return {"results": found}
        return {"error": "no definition of that name in the repository", "name": name}

    def line_range(self, path, start, end):
        """Validated (path, start, end) for a 1-based inclusive range, or an error message."""
        lines = self.builder.file_lines.get(path) if isinstance(path, str) else None
        if lines is None:
            return None, "path is not a file of the repository (the file being completed is not available)"
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
            return None, f"need 1 <= start_line <= end_line <= {len(lines)}"
        if end - start + 1 > self.limits.read_lines:
            return None, f"at most {self.limits.read_lines} lines at a time"
        return (path, start, end), None

    def read_file(self, path, start_line, end_line) -> dict:
        checked, error = self.line_range(path, start_line, end_line)
        if error:
            return {"error": error}
        path, start, end = checked
        return {"path": path, "start_line": start, "end_line": end,
                "text": self._clip("\n".join(self.builder.file_lines[path][start - 1:end]))}

    def execute(self, action: dict) -> dict:
        name = action.get("action")
        if name == "search_code":
            return self.search_code(action.get("query"))
        if name == "find_definition":
            return self.find_definition(action.get("name"))
        if name == "find_usages" and self.limits.edit_mode == "swap":
            return self.find_usages(action.get("name"))
        if name == "file_outline" and self.limits.edit_mode == "swap":
            return self.file_outline(action.get("path"))
        if name == "read_file":
            return self.read_file(action.get("path"), action.get("start_line"), action.get("end_line"))
        return {"error": "unknown action; use search_code, find_definition, read_file or submit_context"}

    def resolve_items(self, items) -> list[Chunk]:
        """Turn items to add into chunks, dropping anything invalid, duplicated or already covered."""
        resolved: list[Chunk] = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            if "chunk" in item:
                chunk_id = item["chunk"]
                if type(chunk_id) is int and chunk_id in self.seen:
                    chunk = self.builder.chunks[chunk_id]
                else:
                    continue
            else:
                checked, error = self.line_range(item.get("path"), item.get("start_line"), item.get("end_line"))
                if error:
                    continue
                path, start, end = checked
                text = "\n".join(self.builder.file_lines[path][start - 1:end])
                chunk = Chunk(path, text, start - 1, end - 1, "range", self.count_tokens(text))
            if any(covers(c, chunk) for c in resolved):
                continue
            resolved.append(chunk)
            if len(resolved) >= self.limits.max_items:
                break
        return resolved

    def apply_edit(self, default_chunks: list[Chunk], add, drop) -> list[Chunk]:
        """The default context with the planner's edit: added chunks first (most relevant), then the default chunks it
        did not drop and that no added chunk already covers. An empty edit returns the default unchanged."""
        if self.limits.edit_mode == "append":
            return self._append_edit(default_chunks, add)
        if self.limits.edit_mode == "swap":
            return self._swap_edit(default_chunks, add, drop)
        default_ids = {self.chunk_id[id(c)] for c in default_chunks}
        dropped = {chunk_id for chunk_id in (drop if isinstance(drop, list) else [])
                   if type(chunk_id) is int and chunk_id in default_ids}
        added = self.resolve_items(add)
        kept = [c for c in default_chunks
                if self.chunk_id[id(c)] not in dropped and not any(covers(a, c) for a in added)]
        # an added chunk that is a default chunk is a promotion: it moves to the front
        return added + [c for c in kept if not any(c is a for a in added)]

    def _append_edit(self, default_chunks: list[Chunk], add) -> list[Chunk]:
        """Append mode: every default chunk keeps its rank; only genuinely new code is added, after them. An addition
        that a default chunk already contains is no change; a default chunk that an addition contains is replaced."""
        added = [a for a in self.resolve_items(add) if not any(covers(c, a) for c in default_chunks)]
        added = added[:self.limits.max_add]
        kept = [c for c in default_chunks if not any(covers(a, c) for a in added)]
        return kept + added


    def _swap_edit(self, default_chunks: list[Chunk], add, drop) -> list[Chunk]:
        """Swap mode: new code goes after the default chunks (lowest priority); the planner may name at most `max_evict`
        default chunks to give up, but only together with an addition (a drop alone changes nothing). An addition that a
        default chunk that stays already contains is no change; a default chunk that an addition contains is replaced."""
        default_ids = {self.chunk_id[id(c)] for c in default_chunks}
        evict = [i for i in (drop if isinstance(drop, list) else []) if type(i) is int and i in default_ids][:self.limits.max_evict]
        staying = [c for c in default_chunks if self.chunk_id[id(c)] not in evict]
        added = [a for a in self.resolve_items(add) if not any(covers(c, a) for c in staying)][:self.limits.max_add]
        if not added:
            return default_chunks
        return [c for c in staying if not any(covers(a, c) for a in added)] + added


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
