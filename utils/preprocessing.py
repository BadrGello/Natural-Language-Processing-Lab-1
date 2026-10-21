"""Shared text preprocessing for all three notebooks.

Keep defaults consistent across classification experiments so model comparisons
are fair. Do not remove stopwords by default: useful signal may be lost.
"""

from __future__ import annotations

import re
import unicodedata

_WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")

def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Normalize case and apostrophes
    text = text.casefold().replace("’", "'")

    # Extract words
    return _WORD_PATTERN.findall(text)


def normalize_text(text: str) -> str:
    """Normalize text using the same tokenizer, with single-space separators."""
    return " ".join(tokenize(text))


def tokenize_many(texts: list[str]) -> list[list[str]]:
    """Tokenize a collection of documents without mixing document boundaries."""
    return [tokenize(text) for text in texts]
