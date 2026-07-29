"""Speech normalization for the transcript branch (REC-342).

The asymmetry these tests protect: removing a real word is far worse than
leaving filler in. Filler costs some retrieval quality; a stripped word makes
the chunk say something its speaker did not, and the citation quotes it back
to the reader. So most of this file is about what must SURVIVE.
"""
from __future__ import annotations

import pytest

from src.ingest.textnorm import normalize_speech, speech_report


# ── Filler goes ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,clean", [
    ("um the marginal cost was close to zero",
     "the marginal cost was close to zero"),
    ("uh these businesses ran at 90% gross margins",
     "these businesses ran at 90% gross margins"),
    ("so uh, we uh looked at the numbers", "so we looked at the numbers"),
    ("ummm that is the biggest question", "that is the biggest question"),
    ("erm, the answer is capex", "the answer is capex"),
])
def test_meaningless_interjections_are_removed(raw: str, clean: str) -> None:
    assert normalize_speech(raw) == clean


def test_you_know_as_filler_is_removed() -> None:
    # The comma the parenthetical sat behind survives. Asserting it away would
    # be asserting cosmetics: what matters is that the filler is gone and no
    # real word went with it.
    assert normalize_speech("it is, you know, a huge number") == \
        "it is, a huge number"


def test_stutters_collapse() -> None:
    assert normalize_speech("the the cloud stack") == "the cloud stack"
    assert normalize_speech("it is is is actually the biggest") == \
        "it is actually the biggest"


def test_filler_then_stutter_both_go() -> None:
    """Order matters: filler first, so 'um the the' leaves 'the the' for the
    stutter rule to see."""
    assert normalize_speech("um the the model") == "the model"


# ── Meaning survives — the half that matters ─────────────────────────────────

def test_you_know_as_a_real_clause_survives() -> None:
    """'do you know', 'if you know' are content, not filler. This is why the
    rule is a lookbehind and not a replace."""
    for s in ("do you know the answer", "if you know the margins",
              "that you know this already", "they did not know"):
        assert "know" in normalize_speech(s)
    assert normalize_speech("do you know the answer") == "do you know the answer"


@pytest.mark.parametrize("word", [
    "like", "right", "okay", "actually", "basically", "sort", "kind",
])
def test_ambiguous_discourse_words_are_left_alone(word: str) -> None:
    """Each has a reading where it IS the content — 'a kind of transformer',
    'the right answer'. A normalizer that strips meaning is worse than the
    filler it removes."""
    s = f"this is the {word} of thing we measure"
    assert word in normalize_speech(s)


def test_real_words_containing_filler_letters_survive() -> None:
    """Word boundaries, not substrings: 'humble' contains 'um', 'huh' contains
    'uh', 'summary' contains 'um'."""
    s = "a humble summary of the uhlan hmm"
    out = normalize_speech(s)
    assert "humble" in out and "summary" in out and "uhlan" in out


def test_numbers_and_units_are_untouched() -> None:
    s = "about $100 a user a year, 3.5 billion users, 80% margins"
    assert normalize_speech(s) == s


def test_a_clean_sentence_is_returned_unchanged() -> None:
    """Edited prose must be a no-op — papers and posts never reach this
    function, but a transcript that happens to be clean must not be edited."""
    s = "The marginal cost of serving software approaches zero at scale."
    assert normalize_speech(s) == s


def test_legitimate_doubling_is_an_accepted_loss() -> None:
    """'had had' collapses. Documented rather than fixed: distinguishing it
    from a speech restart needs parsing, and it is vanishingly rare in a
    lecture transcript compared to the restarts."""
    assert normalize_speech("he had had enough") == "he had enough"


# ── Punctuation left behind ──────────────────────────────────────────────────

def test_removal_does_not_leave_dangling_punctuation() -> None:
    assert normalize_speech("well, um, the answer") == "well, the answer"
    assert normalize_speech("um. the answer") == "the answer"


def test_empty_and_whitespace_are_safe() -> None:
    assert normalize_speech("") == ""
    assert normalize_speech("   ") == ""
    assert normalize_speech("um uh erm") == ""


# ── The report ───────────────────────────────────────────────────────────────

def test_the_report_counts_what_would_be_removed() -> None:
    """Used in the ingest log so a mangled source is visible at the time,
    rather than found a month later with an ad-hoc script."""
    r = speech_report("um the the cost is, you know, uh low")
    assert r["filler"] >= 2 and r["stutters"] >= 1 and r["you_know"] == 1
    assert r["words_after"] < r["words_before"]


def test_the_report_on_clean_prose_shows_nothing_to_do() -> None:
    r = speech_report("Marginal cost approaches zero at scale.")
    assert r["filler"] == 0 and r["stutters"] == 0 and r["you_know"] == 0
    assert r["words_after"] == r["words_before"]
