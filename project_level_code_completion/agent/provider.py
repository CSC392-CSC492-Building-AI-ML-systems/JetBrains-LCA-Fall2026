from typing import Callable, Optional

from agent.loop import LLMReply, plan, task_message
from agent.tools import Limits, RepoTools
from data_classes.datapoint_base import DatapointBase
from rag.builder import PREFIX_CAP_TOKENS, RagContextBuilder, crop_prefix


def assemble_context(builder: RagContextBuilder, chunks: list, prefix: str, budget: int, count_tokens: Callable,
                     max_context_tokens: Optional[int] = None, continuation_lines: int = 0) -> tuple[str, list]:
    """Prompt context from the planner's chunks, in the layout RAG uses (best chunk nearest the prefix).

    Chunks are taken in the planner's order; one that does not fit the remaining tokens is skipped, never cut.
    Chunks given by id get the same continuation lines as RAG F; explicit line ranges are used as chosen.
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


def make_agent_provider_factory(count_tokens: Callable[[str], int], budget: int, llm: Callable[[list], LLMReply],
                                max_context_tokens: Optional[int] = None, prefix_cap: int = PREFIX_CAP_TOKENS,
                                limits: Limits = Limits(), planner_prefix_tokens: int = 1500, max_listed_files: int = 100,
                                continuation_lines: int = 20, fallback_options: Optional[dict] = None,
                                include_non_py: bool = True, extension: str = "", lang_sep: str = "",
                                meta_sep: str = "METASEP\n"):
    """Return `factory(datapoint) -> provider` for `evaluate_generation`, like `make_rag_provider_factory`.

    For each target line the planner sees only the file prefix and the repository (never the target line), and its
    submitted chunks become the context. When it submits nothing usable the line falls back to RAG's own context
    (`fallback_options` are the `context_for` options of RAG F) and is flagged `fallback` so the rate is reported.
    """
    fallback_options = {"continuation_lines": continuation_lines, **(fallback_options or {})}

    def factory(datapoint: DatapointBase):
        files = datapoint.get_context()
        if not include_non_py:
            files = {path: code for path, code in files.items() if path.endswith(".py")}
        completion_path = next(iter(datapoint.get_completion()))
        builder = RagContextBuilder(files, completion_path, datapoint.repo_name, count_tokens, extension=extension,
                                    lang_sep=lang_sep, meta_sep=meta_sep)
        listed = sorted(builder.file_lines, key=lambda p: (-builder._path_proximity(p), p))[:max_listed_files]

        def provider(dp: DatapointBase, line_num: int):
            prefix = crop_prefix(dp.get_prefix(line_num), prefix_cap, count_tokens)
            tools = RepoTools(builder, limits, count_tokens)
            planned = plan(task_message(completion_path, crop_prefix(prefix, planner_prefix_tokens, count_tokens),
                                        listed), tools, llm, limits)
            chunks = tools.resolve_items(planned.items)
            context, used = assemble_context(builder, chunks, prefix, budget, count_tokens, max_context_tokens,
                                             continuation_lines)
            fallback = not used
            if fallback:
                result = builder.context_for(prefix, budget=budget, prefix_cap=prefix_cap,
                                             max_context_tokens=max_context_tokens, **fallback_options)
                context, used = result.context, result.chunks
            info = {
                "paths": [chunk.path for chunk in used],
                "lines": [[chunk.start_line, chunk.end_line] for chunk in used],
                "fallback": fallback, "planner_status": planned.status, "steps": planned.steps,
                "tool_calls": planned.tool_calls, "planner_input_tokens": planned.input_tokens,
                "planner_output_tokens": planned.output_tokens, "planner_ms": round(planned.planner_ms, 1),
            }
            return context + "\n" + prefix, dp.get_line(line_num), info

        return provider

    return factory
