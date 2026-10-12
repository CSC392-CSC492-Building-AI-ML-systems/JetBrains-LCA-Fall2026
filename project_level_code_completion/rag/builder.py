from dataclasses import dataclass, field
from typing import Callable, Optional

from composers.path_distance_composer import PathDistanceComposer
from rag.bm25 import BM25Index, tokenize_code
from rag.chunker import Chunk, chunk_repo

MAX_CHUNKS = 30
QUERY_LINES = 10
PREFIX_CAP_TOKENS = 4000


@dataclass
class RagResult:
    context: str  # everything that goes before the prefix
    prefix: str  # the (possibly line-cropped) completion-file prefix
    chunks: list[Chunk] = field(default_factory=list)  # selected, best first
    scores: list[float] = field(default_factory=list)
    fallback: bool = False  # True when the query was empty and path-distance ordering was used


def make_query(prefix: str, n_lines: int = QUERY_LINES) -> str:
    """The last `n_lines` non-empty lines of the prefix. The target line is never part of the prefix."""
    lines = [line for line in prefix.split("\n") if line.strip()]
    return "\n".join(lines[-n_lines:])


def crop_prefix(prefix: str, max_tokens: int, count_tokens: Callable[[str], int]) -> str:
    """Drop leading lines until the prefix fits `max_tokens` (the model-side left crop still applies afterwards)."""
    if len(prefix) <= max_tokens:  # a token is at least one character, so this cannot exceed the cap
        return prefix
    lines = prefix.split("\n")
    lo, hi = 0, len(lines) - 1  # smallest start index whose suffix fits; the last line is always kept
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens("\n".join(lines[mid:])) <= max_tokens:
            hi = mid
        else:
            lo = mid + 1
    return "\n".join(lines[lo:])


class RagContextBuilder:
    """BM25 retrieval over one repository snapshot. Build once per datapoint, call `context_for` per target line."""

    def __init__(self, files: dict[str, str], completion_path: str, repo_name: str,
                 count_tokens: Callable[[str], int], parse: Optional[Callable] = None,
                 extension: str = "", lang_sep: str = "", meta_sep: str = "METASEP\n"):
        self.completion_path = completion_path
        self.repo_name = repo_name
        self.count_tokens = count_tokens
        self.extension, self.lang_sep, self.meta_sep = extension, lang_sep, meta_sep
        # The completion file is excluded: its own prefix is already in the prompt.
        self.chunks = chunk_repo(files, count_tokens, parse, exclude_paths=[completion_path])
        self.index = BM25Index([tokenize_code(chunk.text) for chunk in self.chunks])
        self._costs: dict[int, int] = {}

    def _format(self, chunk: Chunk) -> str:
        return chunk.path + self.meta_sep + chunk.text

    def _cost(self, chunk_id: int) -> int:
        """Token cost of a chunk as it appears in the prompt; cached because chunks recur across target lines."""
        if chunk_id not in self._costs:
            chunk = self.chunks[chunk_id]
            self._costs[chunk_id] = self.count_tokens(self._format(chunk) + self.lang_sep)
        return self._costs[chunk_id]

    def _header(self) -> str:
        return f"{self.extension}{self.lang_sep}{self.repo_name}{self.meta_sep}"

    def _rank(self, query: str) -> tuple[list[tuple[int, float]], bool]:
        if query.strip():
            return self.index.rank(tokenize_code(query)), False
        by_distance = sorted(
            range(len(self.chunks)),
            key=lambda i: (PathDistanceComposer._path_distance(self.completion_path, self.chunks[i].path), i),
        )
        return [(i, 0.0) for i in by_distance], True

    def context_for(self, prefix: str, budget: int, prefix_cap: int = PREFIX_CAP_TOKENS,
                    query_lines: int = QUERY_LINES, max_chunks: int = MAX_CHUNKS) -> RagResult:
        """Context for one target line whose preceding lines are `prefix`.

        `budget` is the total input cap in tokens (context + newline + prefix). Chunks are never truncated:
        one that does not fit the remaining budget is skipped and a smaller one is tried.
        The returned context lists chunks in ascending relevance, so the best one is nearest the prefix.
        """
        prefix = crop_prefix(prefix, prefix_cap, self.count_tokens)
        tail = self.completion_path + self.meta_sep
        remaining = budget - self.count_tokens(self._header() + tail) - self.count_tokens("\n" + prefix)

        ranked, fallback = self._rank(make_query(prefix, query_lines))
        selected: list[tuple[Chunk, float]] = []
        for chunk_id, score in ranked:
            if len(selected) >= max_chunks or remaining <= 0:
                break
            chunk = self.chunks[chunk_id]
            cost = self._cost(chunk_id)
            if cost <= remaining:
                selected.append((chunk, score))
                remaining -= cost

        def assemble() -> str:
            pieces = [self._format(chunk) for chunk, _ in reversed(selected)] + [tail]
            return self._header() + self.lang_sep.join(pieces)

        # Per-chunk costs are only approximately additive for real tokenizers, so verify the exact total and
        # drop the lowest-ranked chunk until it fits.
        context = assemble()
        while selected and self.count_tokens(context + "\n" + prefix) > budget:
            selected.pop()
            context = assemble()
        return RagResult(context=context, prefix=prefix, chunks=[c for c, _ in selected],
                         scores=[s for _, s in selected], fallback=fallback)
