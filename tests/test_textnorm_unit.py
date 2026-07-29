"""Unit tests for PDF text normalization (REC-338).

Every case here is drawn from text actually observed in the indexed corpus —
the ligature and hyphenation examples are real strings from the Chollet, CLIP
and BERT papers, not invented ones.
"""
from __future__ import annotations

from src.ingest.textnorm import (dehyphenate, expand_ligatures, normalize_page,
                                 unwrap_lines)


# ── Ligatures ─────────────────────────────────────────────────────────────────

def test_the_fi_ligature_becomes_fi() -> None:
    # The exact string that made an eval query ungradeable: the Chollet paper
    # indexes as "efﬁciency" (ef + U+FB01 + ciency).
    assert expand_ligatures("skill-acquisition efﬁciency") == \
        "skill-acquisition efficiency"


def test_every_latin_ligature_expands() -> None:
    got = expand_ligatures("ﬀ ﬁ ﬂ ﬃ ﬄ")
    assert got == "ff fi fl ffi ffl"


def test_ordinary_text_is_untouched() -> None:
    s = "efficiency of the flexible workflow"
    assert expand_ligatures(s) == s


def test_superscripts_and_fractions_survive() -> None:
    """The reason this is a targeted map and not NFKC: NFKC would render
    these as '102' and '1/2', silently corrupting a paper's numbers."""
    s = "a 10² speedup on ½ the data"
    assert normalize_page(s) == s
    assert "²" in normalize_page(s)


# ── De-hyphenation ────────────────────────────────────────────────────────────

def test_a_syllable_break_is_rejoined() -> None:
    text = "measuring eﬃciency\n\nthe effi-\nciency of the method"
    assert "efficiency of the method" in normalize_page(text)


def test_a_real_compound_keeps_its_hyphen() -> None:
    """`skill-acquisition` split at its own hyphen must not become one word.
    The hyphenated form appears elsewhere in the document, which is the
    evidence used."""
    text = ("intelligence as skill-acquisition efficiency\n\n"
            "abilities that enable skill-\nacquisition in a domain")
    out = normalize_page(text)
    assert "skill-acquisition in a domain" in out
    assert "skillacquisition" not in out


def test_an_unknown_split_defaults_to_joining() -> None:
    """Neither form seen elsewhere — a syllable break is the commoner case."""
    out = dehyphenate("the gener-\nalization gap")
    assert "generalization gap" in out


def test_joining_wins_when_the_joined_form_is_attested() -> None:
    text = "generalization is hard\n\nthe gener-\nalization gap"
    assert "generalization gap" in dehyphenate(text)


def test_hyphens_not_at_a_line_break_are_left_alone() -> None:
    s = "a well-known state-of-the-art result"
    assert dehyphenate(s) == s


# ── Line unwrapping ───────────────────────────────────────────────────────────

def test_soft_line_breaks_become_spaces() -> None:
    assert unwrap_lines("a sentence broken\nacross two lines") == \
        "a sentence broken across two lines"


def test_paragraph_breaks_survive() -> None:
    """The chunker splits paragraphs on blank lines — destroying them would
    destroy chunk structure."""
    out = unwrap_lines("first para\nwrapped\n\nsecond para")
    assert out == "first para wrapped\n\nsecond para"


# ── The pipeline, end to end ──────────────────────────────────────────────────

def test_a_realistic_page_comes_out_as_prose() -> None:
    page = ("We deﬁne intelligence as skill-acquisition eﬃciency,\n"
            "which diﬀers from the classical view.\n\n"
            "This deﬁ-\nnition has consequences for the eval-\nuation of\n"
            "artiﬁcial systems.")  # ligatures as pymupdf emits them
    out = normalize_page(page)
    assert "ﬁ" not in out and "ﬀ" not in out and "ﬃ" not in out
    assert "definition has consequences" in out
    assert "evaluation of artificial systems" in out
    assert "differs from the classical view" in out
    # Paragraph structure preserved for the chunker.
    assert out.count("\n\n") == 1


def test_normalization_is_deterministic() -> None:
    page = "eﬃcient sys-\ntems and their beneﬁts\nacross lines"
    assert normalize_page(page) == normalize_page(page)
