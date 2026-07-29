"""Turning PDF typography back into text.

A typeset PDF stores what the page LOOKS like, and pymupdf faithfully returns
that. Three artefacts of typesetting survive extraction and are not text:

  * ligatures — "efficiency" is drawn with one glyph for "fi" and extracts as
    "efﬁciency" (U+FB01). Measured at 69% of chunks across the paper corpus.
  * hyphenated line breaks — "efﬁ-\\nciency", a syllable split invented by the
    typesetter to justify a line. 47% of chunks.
  * hard line breaks mid-sentence — a column of fragments rather than prose.

All three reach the embedding model, which tokenizes "efﬁciency" as something
other than the word it is. This module puts the text back before chunking.

Everything here is a pure function of the page text: same PDF in, same text
out, so chunk boundaries and the point ids derived from them stay
deterministic (Epic 5 leans on that).
"""
from __future__ import annotations

import re

# Latin ligature block, expanded explicitly.
#
# NOT unicodedata.normalize("NFKC", text), which would do this AND flatten
# superscripts and fractions: "10²" becomes "102" and "½" becomes "1⁄2".
# In a scaling-laws paper or a protein paper that silently corrupts the
# numbers — the exact failure mode nobody notices until a citation is wrong.
# Keep this list targeted.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "Ĳ": "IJ", "ĳ": "ij", "Œ": "OE", "œ": "oe",
}
_LIG_RE = re.compile("|".join(map(re.escape, _LIGATURES)))

# A word split across a line break: "efﬁ-\nciency". The hyphen is followed by
# the newline; both halves are letters.
_HYPHEN_BREAK = re.compile(r"(\w+)-\n(\w+)")

# Line break that is NOT a paragraph boundary (paragraph = blank line).
_SOFT_BREAK = re.compile(r"(?<!\n)\n(?!\n)")

_WORD = re.compile(r"[A-Za-z][A-Za-z-]*")


def expand_ligatures(text: str) -> str:
    return _LIG_RE.sub(lambda m: _LIGATURES[m.group(0)], text)


def _vocabulary(text: str) -> set[str]:
    """Words the document uses, lowercased, excluding the line-break-hyphen
    fragments themselves — those are the ambiguity we are trying to resolve,
    so they must not vote on it."""
    clean = _HYPHEN_BREAK.sub(" ", text)
    return {w.lower() for w in _WORD.findall(clean)}


def dehyphenate(text: str) -> str:
    """Rejoin words the typesetter split across a line break.

    The ambiguity: "efﬁ-\\nciency" must become "efficiency", but
    "skill-\\nacquisition" must NOT become "skillacquisition" — that one is a
    real compound that happened to break at its own hyphen.

    Resolved with a document-scoped vocabulary rather than a dictionary (no
    dependency, no network, deterministic): if the joined form appears
    elsewhere in this document, join; if the hyphenated form appears
    elsewhere, keep the hyphen; otherwise join, because a syllable break is
    the commoner case by far.
    """
    vocab = _vocabulary(text)

    def fix(m: re.Match) -> str:
        head, tail = m.group(1), m.group(2)
        joined, hyphenated = f"{head}{tail}".lower(), f"{head}-{tail}".lower()
        if joined in vocab:
            return f"{head}{tail}"
        if hyphenated in vocab:
            return f"{head}-{tail}"
        return f"{head}{tail}"

    return _HYPHEN_BREAK.sub(fix, text)


def unwrap_lines(text: str) -> str:
    """Single newlines are typographic, not semantic — collapse them to spaces
    so a chunk reads as prose. Blank lines survive: the chunker splits
    paragraphs on them."""
    return _SOFT_BREAK.sub(" ", text)


def normalize_page(text: str) -> str:
    """The full pipeline for one extracted page, in dependency order:
    ligatures first (so "efﬁ-\\nciency" is already "effi-\\nciency" and its
    joined form can match the vocabulary), then de-hyphenation (which needs
    the newlines), then unwrapping (which removes them)."""
    return unwrap_lines(dehyphenate(expand_ligatures(text)))
