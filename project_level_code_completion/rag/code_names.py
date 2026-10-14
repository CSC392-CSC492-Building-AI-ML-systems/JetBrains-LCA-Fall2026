"""Identifier and definition extraction shared by retrieval (`rag`) and offline analysis (`eval.proxy_metrics`)."""
import keyword
import re

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEFINITION = re.compile(r"^[ \t]*(?:async[ \t]+def|def|class)[ \t]+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)[ \t]*(?::[^=\n]+)?=(?!=)", re.MULTILINE)
KEYWORDS = frozenset(keyword.kwlist)


def identifiers(text: str) -> set[str]:
    """Identifier names (whole, case-sensitive); keywords, dunder names and 1-character names are dropped."""
    return {name for name in IDENTIFIER.findall(text)
            if len(name) > 1 and name not in KEYWORDS and not (name.startswith("__") and name.endswith("__"))}


def definitions(text: str) -> set[str]:
    """Names defined by `def`, `class` or a top-level assignment in `text`."""
    return (set(DEFINITION.findall(text)) | set(ASSIGNMENT.findall(text))) - KEYWORDS
