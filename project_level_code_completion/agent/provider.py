from typing import Callable, Optional

from agent.loop import LLMReply, plan, task_message
from agent.tools import Limits, RepoTools
from data_classes.datapoint_base import DatapointBase
from rag.builder import RagContextBuilder, crop_prefix

BUILDER_OPTIONS = ("include_non_py", "retriever", "embedder", "pooling", "embedding_cache", "chunking", "fine_window",
                   "fine_stride")


def assemble_context(builder: RagContextBuilder, chunks: list, prefix: str, budget: int, count_tokens: Callable,
                     max_context_tokens: Optional[int] = None, continuation_lines: int = 0) -> tuple[str, list]:
    """Prompt context from an ordered chunk list, in the layout RAG uses (first chunk nearest the prefix).

    Chunks are taken in order; one that does not fit the remaining tokens is skipped, never cut. Chunks that come from
    the index get the same continuation lines as RAG F; explicit line ranges are used as chosen.
    """
    tail = builder.completion_path + builder.meta_sep
    remaining = budget - count_tokens(builder._header() + tail) - count_tokens("\n" + prefix)
    if max_context_tokens is not None:
        remaining = min(remaining, max_context_tokens)
    selected = []
    for chunk in chunks:
        extra = 0 if chunk.kind == "range" else continuation_lines
        cost = count_tokens(builder._format(chunk, extra) + builder.lang_sep)
        if cost <= remaining:
            selected.append((chunk, extra))
            remaining -= cost

    def build() -> str:
        pieces = [builder._format(chunk, extra) for chunk, extra in reversed(selected)] + [tail]
        return builder._header() + builder.lang_sep.join(pieces)

    context = build()
    while selected and count_tokens(context + "\n" + prefix) > budget:  # per-chunk costs are only nearly additive
        selected.pop()
        context = build()
    return context, [chunk for chunk, _ in selected]


def fit_additions(builder: RagContextBuilder, kept: list, added: list, prefix: str, budget: int, count_tokens: Callable,
                  max_context_tokens: Optional[int], continuation_lines: int, max_evict: int) -> Optional[tuple[str, list]]:
    """Append mode: `kept` (the default chunks, in rank order) followed by `added`. If an addition does not fit, evict
    the lowest-ranked kept chunks, at most `max_evict`, until every addition fits. None when they cannot all fit."""
    for evicted in range(min(max_evict, len(kept)) + 1):
        candidate = kept[:len(kept) - evicted] + added
        context, used = assemble_context(builder, candidate, prefix, budget, count_tokens, max_context_tokens,
                                         continuation_lines)
        if all(any(chunk is u for u in used) for chunk in added):
            return context, used
    return None


def make_agent_provider_factory(count_tokens: Callable[[str], int], budget: int, llm: Callable[[list], LLMReply],
                                rag_options: Optional[dict] = None, limits: Limits = Limits(),
                                planner_prefix_tokens: int = 1500, max_listed_files: int = 100, extension: str = "",
                                lang_sep: str = "", meta_sep: str = "METASEP\n"):
    """Return `factory(datapoint) -> provider` for `evaluate_generation`, like `make_rag_provider_factory`.

    `rag_options` are the options RAG would run with (`rag=F` plus `context_cap` as `max_context_tokens`). For each
    target line the default context is exactly what that RAG run would build. The planner sees the file prefix and the
    default context (never the target line), and may add repository code, drop default chunks or change nothing.
    Whenever it changes nothing, or fails, the prompt is the RAG prompt, byte for byte; failures are flagged `fallback`.
    """
    options = dict(rag_options or {})
    builder_options = {k: options.pop(k) for k in BUILDER_OPTIONS if k in options}
    if builder_options.get("retriever", "bm25") != "bm25" or builder_options.get("embedder"):
        raise ValueError("the planner agent is built on the BM25 retriever")
    if options.get("order", "best_last") != "best_last":
        raise ValueError("the planner agent needs order=best_last")
    continuation_lines = options.get("continuation_lines", 0)
    max_context_tokens = options.get("max_context_tokens")

    def factory(datapoint: DatapointBase):
        files = datapoint.get_context()
        if not builder_options.get("include_non_py", True):
            files = {path: code for path, code in files.items() if path.endswith(".py")}
        completion_path = next(iter(datapoint.get_completion()))
        builder = RagContextBuilder(files, completion_path, datapoint.repo_name, count_tokens, extension=extension,
                                    lang_sep=lang_sep, meta_sep=meta_sep, chunking=builder_options.get("chunking", "structure"),
                                    fine_window=builder_options.get("fine_window", 10),
                                    fine_stride=builder_options.get("fine_stride", 5))
        listed = sorted(builder.file_lines, key=lambda p: (-builder._path_proximity(p), p))[:max_listed_files]

        def provider(dp: DatapointBase, line_num: int):
            default = builder.context_for(dp.get_prefix(line_num), budget=budget, **options)
            tools = RepoTools(builder, limits, count_tokens)
            task = task_message(completion_path, crop_prefix(default.prefix, planner_prefix_tokens, count_tokens),
                                listed, tools.seed_view(default.chunks))
            planned = plan(task, tools, llm, limits)
            context, used, fallback, changed = default.context, default.chunks, planned.status != "submitted", False
            if not fallback:
                final = tools.apply_edit(default.chunks, planned.add, planned.drop)
                if [id(c) for c in final] != [id(c) for c in default.chunks]:
                    if limits.edit_mode == "append":
                        n_kept = sum(1 for c in final if any(c is d for d in default.chunks))
                        fitted = fit_additions(builder, final[:n_kept], final[n_kept:], default.prefix, budget,
                                               count_tokens, max_context_tokens, continuation_lines, limits.max_evict)
                        edited, edited_used = fitted if fitted else (default.context, default.chunks)
                    else:
                        edited, edited_used = assemble_context(builder, final, default.prefix, budget, count_tokens,
                                                               max_context_tokens, continuation_lines)
                    # dropping every chunk is a valid choice: an empty context; in append mode nothing is ever dropped
                    if edited_used or not final or limits.edit_mode == "append":
                        # an edit whose additions do not fit can leave exactly the RAG context: that is no change
                        changed = [id(c) for c in edited_used] != [id(c) for c in default.chunks]
                        context, used = (edited, edited_used) if changed else (default.context, default.chunks)
                    else:
                        fallback = True  # the edit left nothing that fits: keep the RAG context
            info = {
                "paths": [chunk.path for chunk in used],
                "lines": [[chunk.start_line, chunk.end_line] for chunk in used],
                "kinds": [chunk.kind for chunk in used],  # "range" = explicit lines (no continuation); else an index chunk
                "fallback": fallback, "changed": changed, "planner_status": planned.status, "steps": planned.steps,
                "tool_calls": planned.tool_calls, "planner_input_tokens": planned.input_tokens,
                "planner_output_tokens": planned.output_tokens, "planner_ms": round(planned.planner_ms, 1),
                "actions": [turn["action"] for turn in planned.trace],  # what the planner did, to replay and analyse
            }
            return context + "\n" + default.prefix, dp.get_line(line_num), info

        return provider

    return factory
