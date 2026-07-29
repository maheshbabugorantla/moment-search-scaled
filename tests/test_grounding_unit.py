"""The grounding guard (REC-315): no locator that was not retrieved.

The threat is specific. A model that has just read "p. 4" in its context will
cheerfully write "as shown on page 7", and a reader has no way to tell the two
apart. The defence is structural rather than textual: locators are built in
code from the retrieved payloads and the model's prose is never parsed for
them. What the model CAN do is emit an out-of-range `[n]`, and that is stripped.

So these tests pin two things — that generated text cannot become a locator,
and that the model is told each source's location in its own vocabulary, since
handing a paper a fake "00:00" is what teaches it that pages have timestamps.
"""
from __future__ import annotations

from src import llm
from src.rag.search import _validate_citations


# ── Invented [n] references are stripped ─────────────────────────────────────

def test_a_reference_to_a_source_that_was_not_retrieved_is_removed() -> None:
    # Only the bracket group goes; the sentence around it is left alone.
    assert _validate_citations("Scaling holds [1] but not forever [9].", 3) == \
        "Scaling holds [1] but not forever ."


def test_a_mixed_group_keeps_only_the_real_ones() -> None:
    assert _validate_citations("Both agree [2, 7].", 3) == "Both agree [2]."


def test_every_valid_reference_survives_untouched() -> None:
    text = "First [1]. Second [2, 3]."
    assert _validate_citations(text, 3) == text


def test_zero_is_not_a_source() -> None:
    """1-indexed. A [0] is the model counting from the wrong end, and it points
    at nothing."""
    assert _validate_citations("Claim [0].", 3) == "Claim ."


# ── Prose is never a locator ─────────────────────────────────────────────────

def test_a_page_number_written_in_prose_changes_no_citation() -> None:
    """The core property, stated as a test: the citation payload is built from
    retrieval before generation happens, so whatever the model writes about
    pages cannot reach it. If this ever fails, someone has started parsing the
    answer text — which is the exact defect REC-315 exists to prevent.
    """
    invented = "The result is on page 7 of the paper, slide 30, at 14:13. [1]"
    assert _validate_citations(invented, 2) == invented, (
        "citation validation must not rewrite prose — it only range-checks [n]")


# ── The model is told where each source is, in that kind's vocabulary ────────

def test_a_paper_source_is_labelled_by_page_not_by_a_fake_timestamp() -> None:
    line = llm._label(1, {"kind": "paper", "where": "p. 4", "image": None,
                          "transcript": "Attention is all you need.",
                          "title": "Transformers"})
    assert "p. 4" in line
    assert "00:00" not in line, "a paper does not have a timestamp"


def test_a_document_excerpt_is_called_text_not_transcript() -> None:
    """A paper chunk was never spoken aloud. Calling it a transcript is a small
    lie that shows up in generated prose as 'as he says on page 4'."""
    line = llm._label(1, {"kind": "post", "where": "§ Why now", "image": None,
                          "transcript": "Open weights caught up.", "title": "Moats"})
    assert 'text: "Open weights caught up."' in line
    assert "transcript" not in line


def test_a_video_moment_keeps_the_word_transcript() -> None:
    line = llm._label(2, {"kind": "video", "where": "14:13", "image": b"x",
                          "transcript": "so attention lets us look back",
                          "title": "A talk"})
    assert "transcript:" in line and "14:13" in line


def test_a_source_with_no_image_says_so() -> None:
    line = llm._label(3, {"kind": "paper", "where": "p. 9", "image": None,
                          "transcript": "…", "title": "T"})
    assert "no image" in line


def test_the_title_rides_along_so_the_model_can_group_by_document() -> None:
    """Rule 3 of the system prompt asks the model to group sources from the
    same document into one paragraph. It can only do that if it can tell which
    ones share a document."""
    line = llm._label(1, {"kind": "paper", "where": "p. 4", "image": None,
                          "transcript": "…", "title": "Attention Is All You Need"})
    assert line.startswith("[1] Attention Is All You Need, p. 4")


def test_the_prompt_forbids_the_model_from_writing_a_locator() -> None:
    """Belt to the structural braces: the payload cannot carry an invented
    locator, but the model can still SAY one in prose, and a reader believes
    prose. The instruction is part of the guard, so it is pinned here."""
    for text in (llm.SYSTEM, llm._intro("q", 3)):
        low = text.lower()
        assert "never write" in low
        assert "page" in low and "timestamp" in low
