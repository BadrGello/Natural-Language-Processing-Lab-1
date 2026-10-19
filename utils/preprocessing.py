"""Shared text preprocessing for all three notebooks.

Keep defaults consistent across classification experiments so model comparisons
are fair. Do not remove stopwords by default: useful signal may be lost.
"""

from __future__ import annotations

import re

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase English text, remove punctuation, and return word tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _WORD_PATTERN.findall(text.lower())


def normalize_text(text: str) -> str:
    """Normalize text using the same tokenizer, with single-space separators."""
    return " ".join(tokenize(text))


def tokenize_many(texts: list[str]) -> list[list[str]]:
    """Tokenize a collection of documents without mixing document boundaries."""
    return [tokenize(text) for text in texts]
