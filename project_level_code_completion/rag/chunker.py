import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

MAX_CHUNK_TOKENS = 512
WINDOW_LINES = 40
WINDOW_STRIDE = 20


@dataclass
class Chunk:
    path: str
    text: str
    start_line: int  # 0-based, inclusive
    end_line: int  # 0-based, inclusive
    kind: str  # function | class_header | module_header | window
    n_tokens: int
    symbol: Optional[str] = None  # name of the function/method/class the chunk defines, if any


def _default_count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ParserUnavailable(RuntimeError):
    """The tree-sitter grammar could not be loaded (not a syntax error in the file being chunked)."""


_PARSER_STATE: dict = {"parser": None, "error": None}


def _default_parse(code: str):
    """Parse with the project's tree-sitter parser. Imported lazily because importing it loads the grammar.

    Raises `ParserUnavailable` if the grammar cannot be loaded, e.g. when `tree-sitter-python` has not been cloned into
    the working directory (see the README). The failure is remembered so that the load is not retried for every file.
    """
    if _PARSER_STATE["error"] is not None:
        raise ParserUnavailable(_PARSER_STATE["error"])
    if _PARSER_STATE["parser"] is None:
        try:
            from tree_sitter_parser.parser import parser
        except Exception as error:  # missing clone, failed build, grammar/library version mismatch
            _PARSER_STATE["error"] = f"{type(error).__name__}: {error}"
            raise ParserUnavailable(_PARSER_STATE["error"]) from error
        _PARSER_STATE["parser"] = parser
    return _PARSER_STATE["parser"].parse(bytes(code, "utf8")).root_node


def _warn_parser_unavailable(reason: str) -> None:
    warnings.warn(
        f"tree-sitter grammar not available ({reason}); Python files are cut into plain line windows instead of "
        f"function/class chunks. Clone tree-sitter-python (tag v0.21.0) into the working directory, see the README.",
        RuntimeWarning, stacklevel=3)


def _end_row(node) -> int:
    row, col = node.end_point
    return row - 1 if col == 0 and row > node.start_point[0] else row


def window_chunks(lines: list[str], path: str, first_line: int, count_tokens: Callable[[str], int],
                  window: int = WINDOW_LINES, stride: int = WINDOW_STRIDE) -> list[Chunk]:
    """Overlapping line windows over `lines`, which start at file line `first_line`."""
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + window, len(lines))
        text = "\n".join(lines[start:end])
        if text.strip():
            chunks.append(Chunk(path, text, first_line + start, first_line + end - 1, "window", count_tokens(text)))
        if end == len(lines):
            break
        start += stride
    return chunks


def _make_chunk(path, lines, start, end, kind, count_tokens, header: Optional[str] = None,
                symbol: Optional[str] = None) -> list[Chunk]:
    text = "\n".join(lines[start:end + 1])
    if header is not None:
        text = header + "\n" + text
    n_tokens = count_tokens(text)
    if n_tokens > MAX_CHUNK_TOKENS:
        windows = window_chunks(lines[start:end + 1], path, start, count_tokens)
        if windows:
            windows[0].symbol = symbol  # the first window holds the signature
        return windows
    return [Chunk(path, text, start, end, kind, n_tokens, symbol)]


def _name_of(node) -> Optional[str]:
    name = node.child_by_field_name("name") if node is not None else None
    return name.text.decode("utf8") if name is not None else None


def _definition_of(node):
    """Return the function/class node inside `node`, looking through decorators."""
    if node.type == "decorated_definition":
        return node.child_by_field_name("definition")
    return node


def chunk_file(path: str, code: str, count_tokens: Callable[[str], int] = _default_count_tokens,
               parse: Optional[Callable] = None) -> list[Chunk]:
    """Split one file into function/method chunks plus header chunks; fall back to line windows.

    Non-Python files, files that fail to parse, and oversized chunks use overlapping line windows.
    """
    lines = code.split("\n")
    if not path.endswith(".py"):
        return window_chunks(lines, path, 0, count_tokens)
    try:
        root = (parse or _default_parse)(code)
        if root.has_error:
            raise ValueError("syntax error")
    except ParserUnavailable as error:
        if not _PARSER_STATE.get("warned"):  # once per process: the same problem would repeat for every file
            _PARSER_STATE["warned"] = True
            _warn_parser_unavailable(str(error))
        return window_chunks(lines, path, 0, count_tokens)
    except Exception:  # a syntax error or other parse failure in this file: fall back quietly
        return window_chunks(lines, path, 0, count_tokens)

    chunks: list[Chunk] = []
    covered: set[int] = set()  # lines belonging to a function/class chunk

    for node in root.children:
        definition = _definition_of(node)
        if definition is None:
            continue
        start, end = node.start_point[0], _end_row(node)
        if definition.type == "function_definition":
            chunks += _make_chunk(path, lines, start, end, "function", count_tokens, symbol=_name_of(definition))
            covered.update(range(start, end + 1))
        elif definition.type == "class_definition":
            class_line = lines[definition.start_point[0]]
            methods = [m for m in definition.child_by_field_name("body").children
                       if (_definition_of(m) is not None and _definition_of(m).type == "function_definition")]
            header_end = (methods[0].start_point[0] - 1) if methods else end
            chunks += _make_chunk(path, lines, start, header_end, "class_header", count_tokens,
                                  symbol=_name_of(definition))
            for method in methods:
                m_start, m_end = method.start_point[0], _end_row(method)
                chunks += _make_chunk(path, lines, m_start, m_end, "function", count_tokens, header=class_line,
                                      symbol=_name_of(_definition_of(method)))
            covered.update(range(start, end + 1))

    # Everything outside functions/classes (imports, constants, top-level statements) is one chunk.
    rest = [(i, line) for i, line in enumerate(lines) if i not in covered]
    text = "\n".join(line for _, line in rest)
    if text.strip():
        module = _make_chunk(path, [line for _, line in rest], 0, len(rest) - 1, "module_header", count_tokens)
        for chunk in module:  # map positions back to real file lines
            chunk.start_line, chunk.end_line = rest[chunk.start_line][0], rest[chunk.end_line][0]
        chunks = module + chunks
    return chunks


def chunk_repo(files: dict[str, str], count_tokens: Callable[[str], int] = _default_count_tokens,
               parse: Optional[Callable] = None, exclude_paths: Iterable[str] = (),
               chunking: str = "structure", fine_window: int = 10, fine_stride: int = 5) -> list[Chunk]:
    """Chunk every file of a snapshot, skipping `exclude_paths` and dropping exact duplicate chunk texts.

    `chunking="structure"` (default) uses function/class/header chunks; `"fine"` cuts every file into overlapping
    windows of `fine_window` lines with stride `fine_stride` (token counts are not computed for these).
    """
    if chunking not in ("structure", "fine"):
        raise ValueError(f"chunking must be structure or fine, got {chunking!r}")
    excluded = set(exclude_paths)
    seen: set[str] = set()
    chunks = []
    for path, code in files.items():
        if path in excluded:
            continue
        if chunking == "fine":
            file_chunks = window_chunks(code.split("\n"), path, 0, lambda text: 0, fine_window, fine_stride)
        else:
            file_chunks = chunk_file(path, code, count_tokens, parse)
        for chunk in file_chunks:
            key = chunk.text.strip()
            if key and key not in seen:
                seen.add(key)
                chunks.append(chunk)
    return chunks
