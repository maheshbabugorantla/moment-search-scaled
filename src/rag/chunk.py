"""Semantic chunking for document sources — one packer, two locator schemes.

The transcript chunker (ingest/transcript.py) groups caption cues by TIME. This
module groups prose by SIZE and keeps a *locator* on every chunk, because that
locator is what a citation will point at:

  * `chunk_pages`    — papers. Locator = page. A chunk MAY span a page break,
    and records the page it STARTS on: "see page 7" must land the reader where
    the passage begins, not where it happens to spill.
  * `chunk_markdown` — posts. Locator = heading anchor. A chunk may NOT span a
    section boundary. The asymmetry is deliberate: a page is a rendering
    accident, so straddling one loses nothing, whereas the anchor IS the
    citation target and a chunk covering two sections has an ambiguous one.
    Long sections split into several chunks sharing the anchor.

Pure functions, no I/O, no config reads at call time — the caller passes the
knobs. Determinism is a hard requirement, not a nicety: chunk boundaries decide
chunk `idx`, `idx` decides the Qdrant point id, and Epic 5's redelivered runs
only overwrite instead of duplicating if the same input always yields the same
ids.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence, TypeVar

_PARA_SPLIT = re.compile(r"\n\s*\n")

_K = TypeVar("_K")  # the locator a unit carries: a page number, a section index


@dataclass(frozen=True)
class Chunk:
    idx: int    # 0-based position in the document's chunk sequence
    page: int   # 1-based page the chunk STARTS on
    text: str


# ── The shared packer ─────────────────────────────────────────────────────────

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


def _explode(blocks: Iterable[tuple[_K, str]], max_chars: int,
             overlap_chars: int) -> list[tuple[_K, str]]:
    """(locator, block of text) -> (locator, paragraph) units.

    Blank-line-separated paragraphs are the atoms; one that alone exceeds
    max_chars is split further, so the packer never has to emit an oversize
    chunk."""
    units: list[tuple[_K, str]] = []
    for key, block in blocks:
        text = (block or "").strip()
        if not text:
            continue  # an image-only page, or a heading with nothing under it
        for para in (p.strip() for p in _PARA_SPLIT.split(text)):
            if not para:
                continue
            if len(para) > max_chars:
                units.extend((key, piece)
                             for piece in _split_long(para, max_chars, overlap_chars))
            else:
                units.append((key, para))
    return units


def _pack(units: Sequence[tuple[_K, str]], max_chars: int) -> list[tuple[_K, str]]:
    """Greedy-pack consecutive units up to max_chars. A packed chunk carries the
    locator of its FIRST unit — the page-break rule in the module docstring."""
    packed: list[tuple[_K, str]] = []
    buf: list[str] = []
    buf_key: _K | None = None

    def _flush() -> None:
        nonlocal buf
        if buf:
            packed.append((buf_key, "\n\n".join(buf)))  # type: ignore[arg-type]
            buf = []

    for key, para in units:
        if buf and sum(len(p) + 2 for p in buf) + len(para) > max_chars:
            _flush()
        if not buf:
            buf_key = key
        buf.append(para)
    _flush()
    return packed


def _merge_tiny_tail(packed: list[tuple[_K, str]],
                     min_chars: int) -> list[tuple[_K, str]]:
    """A tiny trailing fragment reads better (and embeds better) merged into its
    predecessor than shipped as a chunk of its own. Greedy packing fills every
    chunk but the last, so the last is the only one that can come out short."""
    if len(packed) > 1 and len(packed[-1][1]) < min_chars:
        _, tail = packed.pop()
        prev_key, prev = packed[-1]
        packed[-1] = (prev_key, f"{prev}\n\n{tail}")
    return packed


# ── Papers: page locators, chunks may cross a page break ─────────────────────

def chunk_pages(pages: Sequence[str], *, max_chars: int = 1400,
                overlap_chars: int = 200, min_chars: int = 80) -> list[Chunk]:
    """Per-page text -> ordered, page-carrying chunks.

    `pages` is one string per page, index 0 = page 1 (parse_pdf's output).
    Pages with no extractable text (scanned/image-only) are skipped, but the
    page numbering marches on — a chunk on the page AFTER a scanned page still
    reports its true page. Same input always yields the same (idx, page, text)
    sequence.
    """
    units = _explode(enumerate(pages, start=1), max_chars, overlap_chars)
    packed = _merge_tiny_tail(_pack(units, max_chars), min_chars)
    return [Chunk(idx=i, page=p, text=t) for i, (p, t) in enumerate(packed)]


# `chunk_markdown` — the anchor-locator counterpart — lands next, on top of the
# same three helpers.
