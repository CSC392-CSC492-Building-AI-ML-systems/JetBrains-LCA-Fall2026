from typing import Callable, Optional

from data_classes.datapoint_base import DatapointBase
from rag.builder import MAX_CHUNKS, PREFIX_CAP_TOKENS, QUERY_LINES, RagContextBuilder


def make_rag_provider_factory(count_tokens: Callable[[str], int], budget: int, prefix_cap: int = PREFIX_CAP_TOKENS,
                              query_lines: int = QUERY_LINES, max_chunks: int = MAX_CHUNKS,
                              include_non_py: bool = True, extension: str = "", lang_sep: str = "",
                              meta_sep: str = "METASEP\n", max_context_tokens: Optional[int] = None,
                              retriever: str = "bm25", embedder=None, pooling: str = "cls",
                              embedding_cache: Optional[str] = None, chunking: str = "structure",
                              fine_window: int = 10, fine_stride: int = 5, **builder_options):
    """Return `factory(datapoint) -> provider` for `evaluate_generation`.

    The factory indexes a datapoint's repo snapshot once; the provider builds the prompt for one target line:
    retrieved context + newline + (cropped) prefix, the same layout the static composers produce.
    `budget` is the total input cap in tokens. The separators should match `composers_config`.
    `embedder` is a Hugging Face model name (loaded once, on first use) or an object with `embed_chunks` and
    `embed_query`; it is only needed when `retriever` is "dense" or "hybrid". Remaining keyword options go to
    `RagContextBuilder.context_for`.
    """
    if retriever != "bm25" and isinstance(embedder, str):
        from rag.dense import DenseEmbedder

        embedder = DenseEmbedder(embedder, pooling=pooling, cache_dir=embedding_cache)

    def factory(datapoint: DatapointBase):
        files = datapoint.get_context()
        if not include_non_py:
            files = {path: code for path, code in files.items() if path.endswith(".py")}
        completion_path = next(iter(datapoint.get_completion()))
        builder = RagContextBuilder(files, completion_path, datapoint.repo_name, count_tokens,
                                    extension=extension, lang_sep=lang_sep, meta_sep=meta_sep, retriever=retriever,
                                    embedder=embedder, chunking=chunking, fine_window=fine_window,
                                    fine_stride=fine_stride)

        def provider(dp: DatapointBase, line_num: int):
            result = builder.context_for(dp.get_prefix(line_num), budget=budget, prefix_cap=prefix_cap,
                                         query_lines=query_lines, max_chunks=max_chunks,
                                         max_context_tokens=max_context_tokens, **builder_options)
            info = {
                "paths": [chunk.path for chunk in result.chunks],
                "lines": [[chunk.start_line, chunk.end_line] for chunk in result.chunks],
                "scores": [round(score, 3) for score in result.scores],
                "fallback": result.fallback,
                "n_definitions": result.n_definitions,
            }
            return result.context + "\n" + result.prefix, dp.get_line(line_num), info

        return provider

    return factory
