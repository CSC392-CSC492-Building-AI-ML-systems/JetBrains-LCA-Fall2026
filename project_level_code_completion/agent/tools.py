"""Read-only repository tools for the planner (RAG_EXPERIMENT_LOG section 8.80).

The tools see exactly what RAG sees: the chunks of `RagContextBuilder`, which never include the completion file.
Lines are numbered from 1 in everything the planner reads or writes.
"""
import json
from dataclasses import dataclass
from typing import Callable

from rag.bm25 import tokenize_code
from rag.builder import RagContextBuilder
from rag.chunker import Chunk


@dataclass(frozen=True)
class Limits:
    tool_calls: int = 5  # tool calls before the planner must submit
    search_results: int = 5
    read_lines: int = 80  # longest range read_file returns or submit_context accepts
    tool_output_tokens: int = 1200  # a tool result is cut to this many tokens, by whole lines
    query_characters: int = 300
    max_items: int = 8  # most items one submit_context may use
    invalid_actions: int = 2  # unusable replies tolerated before the run is abandoned


class RepoTools:
    def __init__(self, builder: RagContextBuilder, limits: Limits, count_tokens: Callable[[str], int]):
        self.builder, self.limits, self.count_tokens = builder, limits, count_tokens
        self.seen: set[int] = set()  # chunk ids the planner has been shown; submit_context may cite only these

    def _clip(self, text: str) -> str:
        lines = text.split("\n")
        while len(lines) > 1 and self.count_tokens("\n".join(lines)) > self.limits.tool_output_tokens:
            lines.pop()
        return "\n".join(lines)

    def _chunk_view(self, chunk_id: int) -> dict:
        chunk = self.builder.chunks[chunk_id]
        self.seen.add(chunk_id)
        return {"chunk": chunk_id, "path": chunk.path, "start_line": chunk.start_line + 1,
                "end_line": chunk.end_line + 1, "symbol": chunk.symbol, "text": self._clip(chunk.text)}

    def search_code(self, query) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > self.limits.query_characters:
            return {"error": f"query must be a non-empty string of at most {self.limits.query_characters} characters"}
        ranked = self.builder.index.rank(tokenize_code(query))[:self.limits.search_results]
        return {"results": [self._chunk_view(chunk_id) for chunk_id, _ in ranked]}

    def find_definition(self, name) -> dict:
        if not isinstance(name, str) or name not in self.builder.symbol_index:
            return {"error": "no function or class with that exact name in the repository", "name": name}
        near = sorted(self.builder.symbol_index[name], key=lambda i: (-self.builder._path_proximity(self.builder.chunks[i].path), i))
        return {"results": [self._chunk_view(chunk_id) for chunk_id in near[:3]]}

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
        """Turn submit_context items into chunks, dropping anything invalid, duplicated or already covered."""
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
            if any(c.path == chunk.path and c.start_line <= chunk.start_line and chunk.end_line <= c.end_line
                   for c in resolved):
                continue
            resolved.append(chunk)
            if len(resolved) >= self.limits.max_items:
                break
        return resolved


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
