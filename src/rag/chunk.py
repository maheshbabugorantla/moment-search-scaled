"""Page-aware semantic chunking — the document counterpart of the transcript
chunker (ingest/transcript.py groups caption cues by TIME; this groups page
text by SIZE, keeping the page as the locator).

Pure functions, no I/O, no config reads at call time — the caller passes the
knobs. Determinism is a hard requirement, not a nicety: chunk boundaries decide
chunk `idx`, `idx` decides the Qdrant point id, and Epic 5's redelivered runs
only overwrite instead of duplicating if the same PDF always yields the same
ids.

The locator decision, written down because a citation will point at it:
a chunk that spans a page break carries the page it STARTS on. "See page 7"
must land the reader where the passage begins, not where it happens to spill.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

_PARA_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    idx: int    # 0-based position in the document's chunk sequence
    page: int   # 1-based page the chunk STARTS on
    text: str


def _split_long(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """A single paragraph longer than max_chars, split at sentence/word
    boundaries with a tail overlap so no thought is cut mid-sentence AND lost."""
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer a sentence boundary, fall back to a word boundary — but
            # only if the cut keeps the piece a meaningful size.
            cut = text.rfind(". ", start, end)
            if cut <= start + max_chars // 2:
                cut = text.rfind(" ", start, end)
            if cut > start + max_chars // 2:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return parts


def chunk_pages(pages: Sequence[str], *, max_chars: int = 1400,
                overlap_chars: int = 200, min_chars: int = 80) -> list[Chunk]:
    """Per-page text -> ordered, page-carrying chunks.

    `pages` is one string per page, index 0 = page 1 (parse_pdf's output).
    Pages with no extractable text (scanned/image-only) are skipped, but the
    page numbering marches on — a chunk on the page AFTER a scanned page still
    reports its true page. Same input always yields the same (idx, page, text)
    sequence.
    """
    # 1. Explode pages into (page, paragraph) units, splitting any paragraph
    #    that alone exceeds max_chars.
    units: list[tuple[int, str]] = []
    for pno, page_text in enumerate(pages, start=1):
        text = (page_text or "").strip()
        if not text:
            continue  # image-only page — nothing to index without OCR
        for para in (p.strip() for p in _PARA_SPLIT.split(text)):
            if not para:
                continue
            if len(para) > max_chars:
                units.extend((pno, piece)
                             for piece in _split_long(para, max_chars, overlap_chars))
            else:
                units.append((pno, para))

    # 2. Greedy-pack consecutive units up to max_chars. The chunk's page is the
    #    page of its FIRST unit (the page-break rule in the module docstring).
    packed: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_page = 0

    def _flush() -> None:
        nonlocal buf
        if buf:
            packed.append((buf_page, "\n\n".join(buf)))
            buf = []

    for pno, para in units:
        if buf and sum(len(p) + 2 for p in buf) + len(para) > max_chars:
            _flush()
        if not buf:
            buf_page = pno
        buf.append(para)
    _flush()

    # 3. A tiny tail fragment reads better (and embeds better) merged into its
    #    predecessor than shipped as a chunk of its own.
    if len(packed) > 1 and len(packed[-1][1]) < min_chars:
        tail_page, tail = packed.pop()
        prev_page, prev = packed[-1]
        packed[-1] = (prev_page, f"{prev}\n\n{tail}")

    return [Chunk(idx=i, page=p, text=t) for i, (p, t) in enumerate(packed)]
