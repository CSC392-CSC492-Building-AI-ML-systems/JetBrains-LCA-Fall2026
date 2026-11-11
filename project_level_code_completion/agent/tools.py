"""Read-only repository tools for the planner (RAG_EXPERIMENT_LOG sections 8.80-8.82).

The tools see exactly what RAG sees: the chunks of `RagContextBuilder`, which never include the completion file.
Lines are numbered from 1 in everything the planner reads or writes. The planner starts from RAG's own context (the
"default") and can only edit it: add repository code, drop default chunks, or change nothing.
"""
import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from rag.bm25 import tokenize_code
from rag.builder import RagContextBuilder
from rag.chunker import Chunk


@dataclass(frozen=True)
class Limits:
    tool_calls: int = 5  # tool calls before the planner must submit
    search_results: int = 5
    read_lines: int = 80  # longest range read_file returns or an added range may have
    tool_output_tokens: int = 1200  # a tool result is cut to this many tokens, by whole lines
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


def covers(outer: Chunk, inner: Chunk) -> bool:
    return outer.path == inner.path and outer.start_line <= inner.start_line and inner.end_line <= outer.end_line


class RepoTools:
    def __init__(self, builder: RagContextBuilder, limits: Limits, count_tokens: Callable[[str], int]):
        self.builder, self.limits, self.count_tokens = builder, limits, count_tokens
        self.seen: set[int] = set()  # chunk ids the planner has been shown; submit_context may cite only these
        self.chunk_id = {id(chunk): i for i, chunk in enumerate(builder.chunks)}

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
                "end_line": chunk.end_line + 1, "symbol": chunk.symbol, "text": self._clip(chunk.text, clip)}

    def seed_view(self, default_chunks: list[Chunk]) -> list[dict]:
        """The default context as the planner sees it: best first, each chunk with its id and rank."""
        return [self._chunk_view(self.chunk_id[id(chunk)], self.limits.seed_chunk_tokens, rank=rank)
                for rank, chunk in enumerate(default_chunks, start=1)]

    def search_code(self, query) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > self.limits.query_characters:
            return {"error": f"query must be a non-empty string of at most {self.limits.query_characters} characters"}
        ranked = self.builder.index.rank(tokenize_code(query))[:self.limits.search_results]
        return {"results": [self._chunk_view(chunk_id) for chunk_id, _ in ranked]}

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


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
