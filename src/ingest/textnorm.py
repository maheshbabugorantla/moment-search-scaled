"""Turning PDF typography back into text.

A typeset PDF stores what the page LOOKS like, and pymupdf faithfully returns
that. What comes back is full of characters that are *presentation forms* —
glyphs chosen by a typesetter — rather than the words a reader (or an
embedding model) should see:

  * ligatures — "efficiency" drawn with one glyph extracts as "efﬁciency"
    (U+FB01). Measured at 69% of chunks across the paper corpus.
  * hyphenated line breaks — "efﬁ-\\nciency", a syllable split invented to
    justify a line. 47% of chunks.
  * hard line breaks mid-sentence, non-breaking spaces, soft hyphens, curly
    quotes, fullwidth and math-styled letters.

All of it reaches the embedding model, which tokenizes "efﬁciency" as
something other than the word it is.

**How the presentation forms are detected.** Not by enumerating characters —
there are thousands, across Latin ligatures, Arabic contextual forms, CJK
compatibility, fullwidth Latin, math alphanumerics and more. Unicode already
records this: every compatibility character carries a decomposition tagged
with the KIND of transformation it represents (`<compat>`, `<font>`, `<wide>`,
`<isolated>`, `<super>`, `<fraction>`, ...). We decompose by tag, which covers
the whole space and stays correct as Unicode grows.

**What is deliberately NOT flattened.** Three tags change meaning rather than
presentation, and blanket NFKC — the obvious one-liner — destroys them:
`<super>` turns "10²" into "102", `<sub>` turns "x₁" into "x1", `<fraction>`
turns "½" into "1⁄2". In a scaling-laws or protein paper that silently
corrupts the numbers, which is the kind of error nobody notices until a
citation is wrong. Everything else decomposes.

Every function here is a pure function of the page text: same PDF in, same
text out, so chunk boundaries and the point ids derived from them stay
deterministic (Epic 5 leans on that).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Decomposition tags whose transformation carries meaning. Everything else is
# presentation and gets expanded. See the module docstring for why this is an
# opt-OUT list rather than an opt-in one.
_MEANINGFUL_TAGS = frozenset({"<super>", "<sub>", "<fraction>"})

# Invisible formatting characters that survive extraction and break
# tokenization: soft hyphen (an invisible hyphenation point), zero-width
# space, word joiner, BOM. ZWJ/ZWNJ are kept — they carry shaping meaning in
# Arabic and Indic scripts, where removing them would change the text.
_ZWJ, _ZWNJ = "‍", "‌"

# Typographic punctuation Unicode does NOT class as compatibility characters,
# because they are distinct characters rather than presentation forms — but a
# PDF uses them where a keyboard would type ASCII. Mapping them makes text
# matchable, and U+2010/2011 in particular are load-bearing: a PDF that
# hyphenates with a real HYPHEN rather than HYPHEN-MINUS would otherwise slip
# past the de-hyphenation rule below.
_PUNCT = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',           # prime, double prime
    "‐": "-", "‑": "-",           # HYPHEN, NON-BREAKING HYPHEN
    "−": "-",                          # MINUS SIGN
}
# En dash and em dash are left alone on purpose: they are punctuation a writer
# chose, not typesetting applied to a word.

_HYPHEN_BREAK = re.compile(r"(\w+)-\n(\w+)")
_SOFT_BREAK = re.compile(r"(?<!\n)\n(?!\n)")
_WORD = re.compile(r"[^\W\d_][\w-]*", re.UNICODE)


def _expand_char(ch: str, _depth: int = 0) -> str:
    """One character -> its semantic text, via Unicode's own decomposition
    data. Recursive because decompositions chain: ㎡ is <square> m + ², and
    the ² it yields must then survive as a superscript."""
    dec = unicodedata.decomposition(ch)
    if not dec or not dec.startswith("<"):
        return ch  # no decomposition, or a canonical one NFC already handled
    tag, _, codes = dec.partition(" ")
    if tag in _MEANINGFUL_TAGS:
        return ch
    expanded = "".join(chr(int(c, 16)) for c in codes.split())
    if _depth >= 4:  # paranoia: Unicode's graph is shallow and acyclic
        return expanded
    return "".join(_expand_char(c, _depth + 1) for c in expanded)


def normalize_unicode(text: str) -> str:
    """Presentation forms -> text, invisibles dropped, typographic punctuation
    folded to ASCII. Superscripts, subscripts and fractions survive."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", text):
        if ch in _PUNCT:
            out.append(_PUNCT[ch])
            continue
        if unicodedata.category(ch) == "Cf" and ch not in (_ZWJ, _ZWNJ):
            continue  # soft hyphen, ZWSP, word joiner, BOM
        out.append(_expand_char(ch))
    return "".join(out)


def _vocabulary(text: str) -> set[str]:
    """Words the document uses, lowercased, excluding the line-break-hyphen
    fragments themselves — those are the ambiguity being resolved, so they
    must not vote on it."""
    return {w.lower() for w in _WORD.findall(_HYPHEN_BREAK.sub(" ", text))}


def dehyphenate(text: str) -> str:
    """Rejoin words the typesetter split across a line break.

    The ambiguity: "effi-\\nciency" must become "efficiency", but
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
        if f"{head}{tail}".lower() in vocab:
            return f"{head}{tail}"
        if f"{head}-{tail}".lower() in vocab:
            return f"{head}-{tail}"
        return f"{head}{tail}"

    return _HYPHEN_BREAK.sub(fix, text)


def unwrap_lines(text: str) -> str:
    """Single newlines are typographic, not semantic — collapse them to spaces
    so a chunk reads as prose. Blank lines survive: the chunker splits
    paragraphs on them."""
    return _SOFT_BREAK.sub(" ", text)


def normalize_page(text: str) -> str:
    """The full pipeline for one extracted page, in dependency order: Unicode
    first (so ligature glyphs become letters and exotic hyphens become ASCII
    ones the next rule can see), then de-hyphenation (which needs the
    newlines), then unwrapping (which removes them)."""
    return unwrap_lines(dehyphenate(normalize_unicode(text)))


def typography_report(text: str) -> dict[str, int]:
    """What typography this text contains, before normalization — so ingest
    can log how mangled a source was instead of us discovering it a month
    later with an ad-hoc script (which is exactly how REC-338 was found).

    Counts characters, plus the line-break hyphenations, by class.
    """
    counts: Counter[str] = Counter()
    for ch in text:
        if ch in _PUNCT:
            counts["smart_punctuation"] += 1
            continue
        if unicodedata.category(ch) == "Cf" and ch not in (_ZWJ, _ZWNJ):
            counts["invisible"] += 1
            continue
        dec = unicodedata.decomposition(ch)
        if dec.startswith("<"):
            tag = dec.partition(" ")[0]
            if tag in _MEANINGFUL_TAGS:
                counts["preserved_semantic"] += 1
            else:
                counts["presentation_form"] += 1
    counts["hyphen_breaks"] = len(_HYPHEN_BREAK.findall(text))
    return dict(counts)
