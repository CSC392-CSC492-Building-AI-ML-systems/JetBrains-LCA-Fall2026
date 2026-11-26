"""Shared lexical tokenizer for bug-localization chunks and query text.

Adapted from the code completion tokenizer. This version filters after
lowercasing and retains a single component (e.g. `name` in `_name`) when it
differs from the whole identifier.
"""

import keyword
import re


TOKENIZER_VERSION = 1
MIN_TOKEN_LENGTH = 2
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
PYTHON_KEYWORDS = frozenset(word.lower() for word in keyword.kwlist)


def tokenizer_identity():
    """Include effective settings, including runtime keyword changes, in caches."""
    return {
        "version": TOKENIZER_VERSION,
        "identifier_pattern": IDENTIFIER.pattern,
        "camel_boundary_pattern": CAMEL_BOUNDARY.pattern,
        "minimum_length": MIN_TOKEN_LENGTH,
        "lowercase": True,
        "keywords": sorted(PYTHON_KEYWORDS),
        "components": "snake-and-camel; omit-components-equal-to-whole",
        "input": "text-only; retain-comments-docstrings-strings",
    }


def tokenize(text: str) -> list[str]:
    """Keep whole ASCII identifiers and their case/snake components, in order.

    Repeated occurrences (including repeated components) retain their frequency.
    Python keywords and tokens shorter than two characters are removed after
    lowercasing. No parsing, stemming, prose stopwords, or path tokens are used.
    """
    tokens = []
    for identifier in IDENTIFIER.findall(text):
        whole = identifier.lower()
        parts = CAMEL_BOUNDARY.sub("_", identifier).split("_")
        candidates = [whole] + [part.lower() for part in parts if part.lower() != whole]
        tokens.extend(term for term in candidates
                      if len(term) >= MIN_TOKEN_LENGTH and term not in PYTHON_KEYWORDS)
    return tokens
