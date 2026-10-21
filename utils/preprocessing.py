"""Shared text preprocessing for all three notebooks.

Pipeline (applied in order):
  1. Unicode normalization (NFKC) — normalises ligatures, full-width chars, etc.
  2. Casefold (aggressive lowercase — handles ß, ﬁ, etc. better than .lower())
  3. Normalize apostrophes to ASCII straight quote
  4. Extract word tokens — keeps contractions intact (e.g. "don't" → ["don't"])
  5. Collapse whitespace / strip

Stop-word removal is intentionally OFF by default so that useful signal
(e.g., negation words like 'not', 'no') is preserved for classification.
It can be toggled via the `remove_stopwords` parameter where supported.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Compiled patterns (module-level for speed)
# ---------------------------------------------------------------------------

# Matches alphanumeric words (removing punctuation and special characters, as required by the lab).
_WORD_PATTERN = re.compile(r"[a-z0-9]+")

# Minimal English stop-word list (NLTK-inspired, opt-in only).
_STOPWORDS: frozenset[str] = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all",
    "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "wasn", "weren",
})


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def tokenize(
    text: str,
    *,
    remove_stopwords: bool = False,
) -> list[str]:
    """Lowercase, normalize Unicode, and return word tokens.

    Contractions are kept intact ("don't" → ["don't"]) rather than split,
    which preserves negation signal important for sentiment/emotion tasks.

    Parameters
    ----------
    text:
        Raw input string.
    remove_stopwords:
        If True, filter out tokens that appear in the built-in stop-word list.
        Default is False so that all notebooks use a common, comparable baseline.

    Returns
    -------
    list[str]
        Ordered list of cleaned word tokens (may be empty for blank input).
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__!r}")

    # Normalize Unicode characters (e.g. full-width letters, ligatures)
    text = unicodedata.normalize("NFKC", text)

    # Casefold is more aggressive than .lower() — handles German ß → ss, etc.
    # Normalize curly/smart apostrophes to ASCII straight quote first.
    text = text.casefold().replace("\u2019", "'").replace("\u2018", "'")

    tokens = _WORD_PATTERN.findall(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    return tokens


def normalize_text(
    text: str,
    *,
    remove_stopwords: bool = False,
) -> str:
    """Normalize text using the same tokenizer, joined with single spaces.

    Useful when a single clean string is needed (e.g., CountVectorizer input).
    """
    return " ".join(tokenize(text, remove_stopwords=remove_stopwords))


def tokenize_many(
    texts: list[str],
    *,
    remove_stopwords: bool = False,
) -> list[list[str]]:
    """Tokenize a collection of documents without mixing document boundaries."""
    return [tokenize(t, remove_stopwords=remove_stopwords) for t in texts]


def get_stopwords() -> frozenset[str]:
    """Return the built-in stop-word set (read-only)."""
    return _STOPWORDS
