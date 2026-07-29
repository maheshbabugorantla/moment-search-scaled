"""Unit tests for PDF text normalization (REC-338).

The ligature and hyphenation examples are real strings from the indexed
corpus (Chollet, CLIP, BERT), not invented ones. The rest exercise the
breadth of Unicode presentation forms the tag-driven expansion is supposed to
cover — and, just as importantly, the three tags it must refuse to touch.
"""
from __future__ import annotations

import pytest

from src.ingest.textnorm import (dehyphenate, normalize_page,
                                 normalize_unicode, typography_report,
                                 unwrap_lines)


# ── Presentation forms expand, whatever script they come from ────────────────

@pytest.mark.parametrize("raw,want", [
    ("skill-acquisition efﬁciency", "skill-acquisition efficiency"),  # the real one
    ("ﬀ ﬁ ﬂ ﬃ ﬄ", "ff fi fl ffi ffl"),        # Latin ligature block
    ("ｆｕｌｌwidth", "fullwidth"),                        # fullwidth Latin
    ("𝐀𝐭𝐭𝐞𝐧𝐭𝐢𝐨𝐧 is all you need", "Attention is all you need"),   # math alphanumerics
    ("section Ⅷ", "section VIII"),                        # Roman numerals
    ("5 ㎡", "5 m²"),                                     # squared unit, superscript kept
    ("call ℡ now", "call TEL now"),                       # symbol glyph
])
def test_presentation_forms_become_text(raw: str, want: str) -> None:
    assert normalize_unicode(raw) == want


def test_arabic_contextual_forms_expand() -> None:
    """Coverage is not Latin-only: the same tag rule handles Arabic
    presentation forms, which a hand-written ligature list would miss."""
    assert normalize_unicode("ﻻ") == "لا"


# ── ...but meaning-bearing forms survive ─────────────────────────────────────

@pytest.mark.parametrize("s", ["10² speedup", "H₂O", "½ the data", "x⁻¹"])
def test_superscripts_subscripts_and_fractions_are_preserved(s: str) -> None:
    """The reason this is tag-driven rather than NFKC: NFKC would render
    these as '102', 'H2O' and '1⁄2', silently corrupting a paper's numbers."""
    assert normalize_page(s) == s


# ── Invisible characters ─────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,want", [
    ("ana­lysis", "analysis"),      # soft hyphen
    ("we​ird", "weird"),            # zero-width space
    ("a b", "a b"),                 # non-breaking space
    ("﻿leading", "leading"),        # BOM
])
def test_invisibles_are_removed(raw: str, want: str) -> None:
    assert normalize_unicode(raw) == want


def test_joiners_are_kept_because_they_carry_meaning() -> None:
    """ZWJ/ZWNJ change how Arabic and Indic text reads — not typography."""
    assert "‍" in normalize_unicode("क्‍ष")


# ── Typographic punctuation ──────────────────────────────────────────────────

def test_smart_quotes_fold_to_ascii() -> None:
    assert normalize_unicode("“quoted” and ’s") == '"quoted" and \'s'


def test_an_exotic_hyphen_still_triggers_dehyphenation() -> None:
    """A PDF hyphenating with U+2010 HYPHEN rather than ASCII would otherwise
    slip past the de-hyphenation rule entirely."""
    assert "efficiency" in normalize_page("effi‐\nciency")


# ── De-hyphenation ───────────────────────────────────────────────────────────

def test_a_syllable_break_is_rejoined() -> None:
    text = "measuring eﬃciency\n\nthe effi-\nciency of the method"
    assert "efficiency of the method" in normalize_page(text)


def test_a_real_compound_keeps_its_hyphen() -> None:
    """`skill-acquisition` split at its own hyphen must not become one word.
    The hyphenated form appearing elsewhere is the evidence used."""
    text = ("intelligence as skill-acquisition efficiency\n\n"
            "abilities that enable skill-\nacquisition in a domain")
    out = normalize_page(text)
    assert "skill-acquisition in a domain" in out
    assert "skillacquisition" not in out


def test_an_unknown_split_defaults_to_joining() -> None:
    assert "generalization gap" in dehyphenate("the gener-\nalization gap")


def test_hyphens_not_at_a_line_break_are_left_alone() -> None:
    s = "a well-known state-of-the-art result"
    assert dehyphenate(s) == s


# ── Line unwrapping ──────────────────────────────────────────────────────────

def test_soft_line_breaks_become_spaces() -> None:
    assert unwrap_lines("a sentence broken\nacross two lines") == \
        "a sentence broken across two lines"


def test_paragraph_breaks_survive() -> None:
    """The chunker splits paragraphs on blank lines — destroying them would
    destroy chunk structure."""
    assert unwrap_lines("first para\nwrapped\n\nsecond para") == \
        "first para wrapped\n\nsecond para"


# ── The pipeline, end to end ─────────────────────────────────────────────────

def test_a_realistic_page_comes_out_as_prose() -> None:
    page = ("We deﬁne intelligence as skill-acquisition eﬃciency,\n"
            "which diﬀers from the classical view.\n\n"
            "This deﬁ-\nnition has consequences for the eval-\nuation of\n"
            "artiﬁcial systems.")
    out = normalize_page(page)
    assert not any(c in out for c in "ﬁﬀﬃ")
    assert "definition has consequences" in out
    assert "evaluation of artificial systems" in out
    assert "differs from the classical view" in out
    assert out.count("\n\n") == 1  # paragraph structure preserved


def test_normalization_is_deterministic() -> None:
    page = "eﬃcient sys-\ntems and their beneﬁts\nacross lines"
    assert normalize_page(page) == normalize_page(page)


def test_clean_text_is_left_exactly_alone() -> None:
    s = "A perfectly ordinary sentence, already clean.\n\nSecond paragraph."
    assert normalize_page(s) == s


# ── The detection report ─────────────────────────────────────────────────────

def test_the_report_classifies_what_it_finds() -> None:
    r = typography_report("efﬁ-\nciency 10² “x” ana­lysis")
    assert r["presentation_form"] == 1     # the ﬁ ligature
    assert r["preserved_semantic"] == 1    # the superscript
    assert r["smart_punctuation"] == 2     # the curly quotes
    assert r["invisible"] == 1             # the soft hyphen
    assert r["hyphen_breaks"] == 1


def test_the_report_is_empty_for_clean_text() -> None:
    r = typography_report("nothing to see here")
    assert not any(r.values())
