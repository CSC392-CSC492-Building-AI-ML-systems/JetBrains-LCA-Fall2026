import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from composers.path_distance_composer import PathDistanceComposer
from rag.bm25 import BM25Index, tokenize_code
from rag.chunker import Chunk, chunk_repo
from rag.code_names import IDENTIFIER, KEYWORDS, definitions

MAX_CHUNKS = 30
QUERY_LINES = 10
PREFIX_CAP_TOKENS = 4000
RRF_CONSTANT = 60  # reciprocal-rank fusion: score = sum over rankings of 1 / (RRF_CONSTANT + rank)
DENSE_CANDIDATES = 200  # how many top chunks of each ranking take part in a hybrid fusion

IMPORT_FROM = re.compile(r"^[ \t]*from[ \t]+([.\w]+)[ \t]+import[ \t]+(.+)$", re.MULTILINE)
IMPORT_PLAIN = re.compile(r"^[ \t]*import[ \t]+(.+)$", re.MULTILINE)


@dataclass
class RagResult:
    context: str  # everything that goes before the prefix
    prefix: str  # the (possibly line-cropped) completion-file prefix
    chunks: list[Chunk] = field(default_factory=list)  # selected, best first
    scores: list[float] = field(default_factory=list)
    fallback: bool = False  # True when the query was empty and path-distance ordering was used
    n_definitions: int = 0  # how many of the first `chunks` came from symbol-definition lookup


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


def parse_imports(code: str) -> tuple[set[str], set[str]]:
    """(module names, imported names) from the import statements of `code` (single-line statements only)."""
    modules: set[str] = set()
    names: set[str] = set()
    for module, imported in IMPORT_FROM.findall(code):
        modules.add(module.lstrip("."))
        for part in imported.strip("() \t").split(","):
            name = part.strip().split(" as ")[0].strip()
            if name and name != "*":
                names.add(name)
    for statement in IMPORT_PLAIN.findall(code):
        for part in statement.split(","):
            module = part.strip().split(" as ")[0].strip()
            if module:
                modules.add(module)
    return {m for m in modules if m}, names


def reciprocal_rank_fusion(rankings: list[list[int]], constant: int = RRF_CONSTANT) -> list[tuple[int, float]]:
    """Fuse several best-first rankings of ids into one list of (id, score), best first; ties keep the lower id."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class RagContextBuilder:
    """Retrieval over one repository snapshot. Build once per datapoint, call `context_for` per target line.

    `retriever` is "bm25" (default), "dense" (cosine similarity of `embedder` vectors) or "hybrid" (reciprocal-rank
    fusion of both). `chunking` is "structure" (functions, classes, headers; default) or "fine" (overlapping windows
    of `fine_window` lines with stride `fine_stride`).
    """

    def __init__(self, files: dict[str, str], completion_path: str, repo_name: str,
                 count_tokens: Callable[[str], int], parse: Optional[Callable] = None,
                 extension: str = "", lang_sep: str = "", meta_sep: str = "METASEP\n",
                 retriever: str = "bm25", embedder=None, chunking: str = "structure",
                 fine_window: int = 10, fine_stride: int = 5):
        if retriever not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"retriever must be bm25, dense or hybrid, got {retriever!r}")
        if retriever != "bm25" and embedder is None:
            raise ValueError(f"retriever={retriever!r} needs an embedder")
        self.completion_path = completion_path
        self.repo_name = repo_name
        self.count_tokens = count_tokens
        self.extension, self.lang_sep, self.meta_sep = extension, lang_sep, meta_sep
        self.retriever, self.embedder = retriever, embedder
        # The completion file is excluded: its own prefix is already in the prompt.
        self.chunks = chunk_repo(files, count_tokens, parse, exclude_paths=[completion_path], chunking=chunking,
                                 fine_window=fine_window, fine_stride=fine_stride)
        self.file_lines = {path: code.split("\n") for path, code in files.items() if path != completion_path}
        self.index = BM25Index([tokenize_code(chunk.text) for chunk in self.chunks])
        self.chunk_vectors = None
        if retriever != "bm25":
            self.chunk_vectors = np.asarray(
                embedder.embed_chunks([f"{chunk.path}\n{chunk.text}" for chunk in self.chunks]), dtype=np.float32)
        self._costs: dict[tuple[int, int], int] = {}
        self._proximity: dict[str, float] = {}
        self.symbol_index: dict[str, list[int]] = {}  # symbol name -> chunks that define it
        for chunk_id, chunk in enumerate(self.chunks):
            if chunk.symbol and not (chunk.symbol.startswith("__") and chunk.symbol.endswith("__")):
                self.symbol_index.setdefault(chunk.symbol, []).append(chunk_id)

    def _format(self, chunk: Chunk, continuation_lines: int = 0) -> str:
        text = chunk.text
        if continuation_lines and chunk.kind != "module_header":
            following = self.file_lines.get(chunk.path, [])[chunk.end_line + 1:chunk.end_line + 1 + continuation_lines]
            if any(line.strip() for line in following):
                text = text + "\n" + "\n".join(following)
        return chunk.path + self.meta_sep + text

    def _cost(self, chunk_id: int, continuation_lines: int = 0) -> int:
        """Token cost of a chunk as it appears in the prompt; cached because chunks recur across target lines."""
        key = (chunk_id, continuation_lines)
        if key not in self._costs:
            self._costs[key] = self.count_tokens(self._format(self.chunks[chunk_id], continuation_lines)
                                                 + self.lang_sep)
        return self._costs[key]

    def _header(self) -> str:
        return f"{self.extension}{self.lang_sep}{self.repo_name}{self.meta_sep}"

    def _path_proximity(self, path: str) -> float:
        """1 / (1 + path distance to the completion file): 1 for the same directory, smaller the farther away."""
        if path not in self._proximity:
            self._proximity[path] = 1.0 / (1.0 + PathDistanceComposer._path_distance(self.completion_path, path))
        return self._proximity[path]

    def _imported_chunk_ids(self, prefix: str) -> set[int]:
        """Chunks of files the completion file imports (module `a.b` matches `.../a/b.py` or `.../a/b/__init__.py`)."""
        modules, _ = parse_imports(prefix)
        suffixes = tuple(s for m in modules for s in ("/" + m.replace(".", "/") + ".py",
                                                      "/" + m.replace(".", "/") + "/__init__.py"))
        if not suffixes:
            return set()
        return {i for i, chunk in enumerate(self.chunks) if ("/" + chunk.path).endswith(suffixes)}

    def _dense_ranking(self, query: str) -> list[tuple[int, float]]:
        """Chunks by cosine similarity to the query, best first (ties keep the lower id)."""
        similarity = self.chunk_vectors @ np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        order = np.argsort(-similarity, kind="stable")
        return [(int(i), float(similarity[i])) for i in order]

    def _rank(self, query: str, prefix: str, proximity_weight: float, import_weight: float
              ) -> tuple[list[tuple[int, float]], bool]:
        if query.strip():
            if self.retriever == "dense":
                return self._dense_ranking(query), False
            ranked = self.index.rank(tokenize_code(query))
            if self.retriever == "hybrid":
                dense = self._dense_ranking(query)[:DENSE_CANDIDATES]
                fused = reciprocal_rank_fusion([[i for i, _ in ranked[:DENSE_CANDIDATES]], [i for i, _ in dense]])
                return fused, False
            if ranked and (proximity_weight or import_weight):
                top = ranked[0][1]
                imported = self._imported_chunk_ids(prefix) if import_weight else set()
                fused = [(chunk_id, score / top
                          + proximity_weight * self._path_proximity(self.chunks[chunk_id].path)
                          + (import_weight if chunk_id in imported else 0.0)) for chunk_id, score in ranked]
                ranked = sorted(fused, key=lambda item: (-item[1], item[0]))
            return ranked, False
        by_distance = sorted(
            range(len(self.chunks)),
            key=lambda i: (PathDistanceComposer._path_distance(self.completion_path, self.chunks[i].path), i),
        )
        return [(i, 0.0) for i in by_distance], True

    def _definition_candidates(self, prefix: str, query: str, use_imports: bool) -> list[tuple[int, float]]:
        """Chunks defining names the query uses (or the file imports) that the prefix does not itself define.

        A name scores its number of occurrences in the query lines, plus 0.5 if the file imports it. Each name
        contributes at most the two defining chunks nearest the completion file.
        """
        used = Counter(n for n in IDENTIFIER.findall(query) if n not in KEYWORDS)
        imported = parse_imports(prefix)[1] if use_imports else set()
        local = definitions(prefix)
        scored = []
        for name in set(used) | imported:
            if name in local or name not in self.symbol_index:
                continue
            nearest = sorted(self.symbol_index[name], key=lambda i: (-self._path_proximity(self.chunks[i].path), i))
            for chunk_id in nearest[:2]:
                scored.append((chunk_id, used[name] + (0.5 if name in imported else 0.0)))
        return sorted(scored, key=lambda item: (-item[1], item[0]))

    def context_for(self, prefix: str, budget: int, prefix_cap: int = PREFIX_CAP_TOKENS,
                    query_lines: int = QUERY_LINES, max_chunks: int = MAX_CHUNKS,
                    max_context_tokens: Optional[int] = None, proximity_weight: float = 0.0,
                    import_weight: float = 0.0, definitions_first: bool = False, definition_chunks: int = 8,
                    definition_tokens: Optional[int] = 2000, definition_imports: bool = True,
                    order: str = "best_last", continuation_lines: int = 0) -> RagResult:
        """Context for one target line whose preceding lines are `prefix`.

        `budget` is the total input cap in tokens (context + newline + prefix). Chunks are never truncated:
        one that does not fit the remaining budget is skipped and a smaller one is tried.
        `max_context_tokens`, if set, also caps the tokens spent on retrieved chunks, so the prompt may be shorter
        than `budget`. With `order="best_last"` (default) the context lists chunks in ascending relevance, so the best
        one is nearest the prefix; `"best_first"` reverses that.

        Optional signals (all off by default, so the default behaviour is plain BM25):
        `proximity_weight` adds `w / (1 + path distance)` to the BM25 score normalised by the best match
        (BM25 retriever only); `import_weight` adds a constant to chunks of files the completion file imports
        (BM25 retriever only); `definitions_first` first selects up to `definition_chunks` chunks (at most
        `definition_tokens` tokens) that define names used in the last `query_lines` lines or imported by the file,
        then fills with the retriever's ranking; `continuation_lines` appends that many lines that follow each chunk
        in its file.
        """
        prefix = crop_prefix(prefix, prefix_cap, self.count_tokens)
        tail = self.completion_path + self.meta_sep
        remaining = budget - self.count_tokens(self._header() + tail) - self.count_tokens("\n" + prefix)
        if max_context_tokens is not None:
            remaining = min(remaining, max_context_tokens)

        query = make_query(prefix, query_lines)
        selected: list[tuple[Chunk, float]] = []
        taken: set[int] = set()
        n_definitions = 0
        if definitions_first:
            def_budget = remaining if definition_tokens is None else min(remaining, definition_tokens)
            for chunk_id, score in self._definition_candidates(prefix, query, definition_imports):
                if n_definitions >= definition_chunks or len(selected) >= max_chunks or chunk_id in taken:
                    continue
                cost = self._cost(chunk_id, continuation_lines)
                if cost <= def_budget:
                    selected.append((self.chunks[chunk_id], score))
                    taken.add(chunk_id)
                    def_budget -= cost
                    remaining -= cost
                    n_definitions += 1

        ranked, fallback = self._rank(query, prefix, proximity_weight, import_weight)
        for chunk_id, score in ranked:
            if len(selected) >= max_chunks or remaining <= 0:
                break
            if chunk_id in taken:
                continue
            cost = self._cost(chunk_id, continuation_lines)
            if cost <= remaining:
                selected.append((self.chunks[chunk_id], score))
                taken.add(chunk_id)
                remaining -= cost

        def assemble() -> str:
            ordered = reversed(selected) if order == "best_last" else selected
            pieces = [self._format(chunk, continuation_lines) for chunk, _ in ordered] + [tail]
            return self._header() + self.lang_sep.join(pieces)

        # Per-chunk costs are only approximately additive for real tokenizers, so verify the exact total and
        # drop the lowest-ranked chunk until it fits.
        context = assemble()
        while selected and self.count_tokens(context + "\n" + prefix) > budget:
            selected.pop()
            n_definitions = min(n_definitions, len(selected))
            context = assemble()
        return RagResult(context=context, prefix=prefix, chunks=[c for c, _ in selected],
                         scores=[s for _, s in selected], fallback=fallback, n_definitions=n_definitions)
