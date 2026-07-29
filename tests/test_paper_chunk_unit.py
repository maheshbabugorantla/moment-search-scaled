"""Unit tests for page-aware parsing and chunking (REC-308).

The invariants a citation will later stand on: every chunk carries a page
within 1..page_count, pages never decrease along the chunk sequence, no chunk
is empty, and the same PDF yields the same chunks every run (Epic 5's
redelivery guarantee keys point ids off chunk idx).

The fixture PDF is generated programmatically with pymupdf at test time — no
binary fixture lives in git.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest.paper import parse_pdf
from src.rag.chunk import Chunk, chunk_pages


def _make_pdf(path: Path, page_texts: list[str | None]) -> Path:
    """One page per entry; None = an image-only page (no extractable text)."""
    import pymupdf

    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text is None:
            # A gray square and no text — what a scanned page looks like.
            page.draw_rect(pymupdf.Rect(72, 72, 200, 200), fill=(0.5, 0.5, 0.5))
        else:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


# ── parse_pdf ─────────────────────────────────────────────────────────────────

def test_parse_pdf_returns_one_string_per_page(tmp_path: Path) -> None:
    path = _make_pdf(tmp_path / "three.pdf", ["alpha", "beta", "gamma"])
    pages = parse_pdf(path)
    assert len(pages) == 3
    for expected, got in zip(["alpha", "beta", "gamma"], pages):
        assert expected in got


def test_an_image_only_page_yields_an_empty_string_not_a_missing_entry(
    tmp_path: Path
) -> None:
    path = _make_pdf(tmp_path / "scanned.pdf", ["text", None, "more text"])
    pages = parse_pdf(path)
    assert len(pages) == 3
    assert not pages[1].strip()


# ── chunk_pages invariants ────────────────────────────────────────────────────

def _para(word: str, n: int = 40) -> str:
    return " ".join([word] * n)


def test_every_chunk_carries_a_valid_starting_page() -> None:
    pages = [_para("one"), _para("two"), _para("three")]
    chunks = chunk_pages(pages)
    assert chunks
    for c in chunks:
        assert 1 <= c.page <= len(pages)
        assert c.text.strip()


def test_pages_are_non_decreasing_across_the_chunk_sequence() -> None:
    pages = [f"{_para(f'w{p}')}\n\n{_para(f'x{p}')}" for p in range(1, 6)]
    chunks = chunk_pages(pages, max_chars=300)
    got = [c.page for c in chunks]
    assert got == sorted(got)


def test_a_chunk_spanning_a_page_break_records_the_page_it_began_on() -> None:
    # Two short paragraphs that pack into ONE chunk despite living on
    # different pages — the chunk must report page 1, where it starts.
    chunks = chunk_pages(["short intro paragraph.", "short continuation."],
                         max_chars=1400)
    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert "continuation" in chunks[0].text


def test_image_only_pages_are_skipped_but_numbering_survives() -> None:
    # Page 2 is scanned (empty text). Page 3's content must still say page 3.
    chunks = chunk_pages([_para("first"), "", _para("third")], max_chars=300)
    assert {c.page for c in chunks} == {1, 3}


def test_chunking_is_deterministic() -> None:
    pages = [f"{_para(f'a{p}')}\n\n{_para(f'b{p}', 80)}" for p in range(1, 4)]
    first = chunk_pages(pages, max_chars=500)
    second = chunk_pages(pages, max_chars=500)
    assert [(c.idx, c.page, c.text) for c in first] == \
           [(c.idx, c.page, c.text) for c in second]
    assert [c.idx for c in first] == list(range(len(first)))


def test_a_long_paragraph_splits_below_max_chars_with_overlap() -> None:
    long_para = _para("token", 400)  # ~2400 chars, no paragraph breaks
    chunks = chunk_pages([long_para], max_chars=500, overlap_chars=100)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 500
        assert c.page == 1


def test_no_chunk_is_ever_empty_and_a_tiny_tail_merges() -> None:
    pages = [_para("body", 100), "tail."]  # the tail alone is < min_chars
    chunks = chunk_pages(pages, max_chars=400, min_chars=80)
    assert all(c.text.strip() for c in chunks)
    assert "tail." in chunks[-1].text  # merged, not shipped as a fragment


def test_the_whole_document_text_is_preserved() -> None:
    """Nothing silently dropped: every paragraph lands in some chunk."""
    paras = [f"paragraph {i} " + _para(f"p{i}", 20) for i in range(8)]
    pages = ["\n\n".join(paras[:4]), "\n\n".join(paras[4:])]
    joined = "\n".join(c.text for c in chunk_pages(pages, max_chars=600))
    for i in range(8):
        assert f"paragraph {i} " in joined


# ── end-to-end: parse then chunk (the REC-308 verify) ─────────────────────────

def test_parse_then_chunk_carries_pages_from_a_real_pdf(tmp_path: Path) -> None:
    path = _make_pdf(tmp_path / "paper.pdf",
                     [f"Section {p}. " + _para(f"page{p}", 30) for p in (1, 2, 3)])
    pages = parse_pdf(path)
    chunks = chunk_pages(pages)
    assert chunks
    assert all(1 <= c.page <= 3 for c in chunks)
    pages_seen = [c.page for c in chunks]
    assert pages_seen == sorted(pages_seen)
