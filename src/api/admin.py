"""Admin surface for non-video sources — papers and decks.

Mirrors the async contract the video path already honours: validate the shape,
insert a `pending` row, return 202, and let a worker do the work. Nothing here
touches the network.

Why shape-only validation. The endpoint must return 202 for a URI it has never
tried to fetch. The SLA probe posts ~30 documents at deliberately fake URIs and
only samples accept-latency on a 202 response; validating reachability here
would turn that gate into a 400 storm and read as an infinite latency weeks
later. Unfetchable URIs surface as `failed` from the worker, which is where the
fetch actually happens.
"""
from __future__ import annotations

import hashlib
import logging
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

from .. import db
from ..config import SOURCE_KINDS, SOURCE_STATUSES
from .errors import UpstreamError
from .videos import require_auth, user_id

log = logging.getLogger("momentsearch.admin")

router = APIRouter(prefix="/admin", tags=["admin"])

# Videos keep their own endpoint; this router is for the kinds that arrive as
# documents. `video` is rejected here on purpose — POST /api/videos owns it.
DOCUMENT_KINDS = tuple(k for k in SOURCE_KINDS if k != "video")

# URI schemes we are willing to record. `storage://` refers to an object already
# uploaded to our bucket; http(s) is fetched by the worker.
ALLOWED_SCHEMES = ("http", "https", "storage")


class DocumentRequest(BaseModel):
    uri: str
    kind: str
    title: str | None = None


def document_id(uri: str) -> str:
    """Deterministic id, so re-posting the same URI updates one row instead of
    accumulating duplicates — the same property `yt_<id>` gives videos."""
    return "doc_" + hashlib.sha256(uri.encode()).hexdigest()[:12]


def _validate(req: DocumentRequest) -> str:
    if req.kind not in DOCUMENT_KINDS:
        raise HTTPException(
            400, f"kind must be one of {', '.join(DOCUMENT_KINDS)} — got {req.kind!r}.")
    uri = req.uri.strip()
    if not uri:
        raise HTTPException(400, "uri is required.")
    parsed = urlparse(uri)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise HTTPException(
            400, f"uri scheme must be one of {', '.join(ALLOWED_SCHEMES)} — got "
                 f"{parsed.scheme or 'none'!r}.")
    if not parsed.netloc and not parsed.path:
        raise HTTPException(400, "uri has no location.")
    return uri


@router.post("/documents", status_code=202, dependencies=[Depends(require_auth)])
def register_document(req: DocumentRequest, uid: str = Depends(user_id)):
    """Accept a paper or a deck. Returns before any parsing happens."""
    uri = _validate(req)
    doc_id = document_id(uri)
    try:
        row = db.upsert_pending_document({
            "id": doc_id, "user_id": uid, "kind": req.kind, "uri": uri,
            "title": req.title,
        })
    except Exception as exc:
        # The manifest write is the one upstream this path genuinely depends
        # on. Its failure is theirs, not ours — 502, and traceable by source id.
        log.error("enqueue failed for %s (%s): %s", doc_id, uri, exc)
        raise UpstreamError(
            "Could not record the source — the manifest database is unavailable.",
            source_id=doc_id) from exc
    return {"id": row["id"], "status": row["status"], "kind": row["kind"]}


# ── Unified status ────────────────────────────────────────────────────────────

# What a source looks like from the outside, whatever its kind. Deliberately
# narrower than the row: this is an admin status view, not a data dump.
_SOURCE_FIELDS = ("id", "kind", "status", "title", "pct", "uri", "error",
                  "created_at", "updated_at")

MAX_PAGE = 500


def _source(row: dict) -> dict:
    out = {k: row.get(k) for k in _SOURCE_FIELDS}
    # `pct` is NOT NULL DEFAULT 0 in the schema, but a row written before the
    # migration backfilled could still surface None through a stale connection.
    out["pct"] = out["pct"] or 0
    return out


@router.get("/sources", dependencies=[Depends(require_auth)])
def list_sources(
    uid: str = Depends(user_id),
    kind: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=MAX_PAGE),
    offset: int = Query(default=0, ge=0),
):
    """One read over the whole manifest, whatever the source kind.

    Reports; it does not measure. `pct` renders whatever the column holds —
    the worker writes it per stage (REC-310), monotone within a run, 100
    exactly when a source reaches `indexed`. `status` remains the terminal
    signal; `pct` says how far a non-terminal run has come.
    """
    if kind is not None and kind not in SOURCE_KINDS:
        raise HTTPException(400, f"kind must be one of {', '.join(SOURCE_KINDS)}.")
    if status is not None and status not in SOURCE_STATUSES:
        raise HTTPException(400, f"status must be one of {', '.join(SOURCE_STATUSES)}.")
    rows = db.list_sources(uid, kind=kind, status=status, limit=limit, offset=offset)
    total = db.count_sources(uid, kind=kind, status=status)
    return {
        "sources": [_source(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
        # Explicit rather than making the poll loop compare arithmetic.
        "next_offset": offset + len(rows) if offset + len(rows) < total else None,
    }


# ── Test fixture ──────────────────────────────────────────────────────────────

_FIXTURE_PAGES = (
    "MomentSearch fixture paper — page one. Retrieval augmented generation "
    "combines a retriever with a generator so answers cite their sources.",
    "Page two. Reciprocal rank fusion merges ranked lists from incomparable "
    "scorers by rank alone, which is why the two branches can disagree on "
    "score scales and still fuse.",
    "Page three. Deterministic chunk identifiers let a redelivered ingest run "
    "overwrite its previous points instead of duplicating them.",
)


@router.get("/fixtures/tiny.pdf")
def fixture_pdf() -> Response:
    """A tiny 3-page PDF, generated per request — the lifecycle contract test
    registers THIS URL so a real end-to-end paper ingest needs no external
    network (arXiv rate limits and MB-scale downloads made the suite flaky).

    Unauthenticated on purpose: the worker's fetch sends no bearer token, and
    the endpoint discloses nothing but lorem-adjacent fixture text.
    """
    import pymupdf

    doc = pymupdf.open()
    for text in _FIXTURE_PAGES:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.set_metadata({})  # keep the bytes as reproducible as pymupdf allows
    body = doc.tobytes()
    doc.close()
    return Response(content=body, media_type="application/pdf")


# Prose before any heading (so `_top` is exercised end to end), a DUPLICATE
# heading (so the -1 suffix is), a bold pseudo-heading, a position-0 banner and
# an in-body chart. Deliberately not a lorem dump:
# the lifecycle test asserts on the anchors and the image verdicts this
# produces. The image URLs are interpolated from the REQUEST's base URL, so the
# worker (which reaches the API as http://api:8000) fetches them from the same
# host it fetched the markdown from — no external network anywhere.
_FIXTURE_POST = """A dateline and standfirst, the way an exported post opens
before its title heading. This prose belongs to no heading, so the parser files
it under the `_top` pseudo-anchor — which every renderer resolves, because it
is the page itself.

![decorative banner]({base}admin/fixtures/banner.png)

# MomentSearch fixture post

The opening paragraph under the title, which carries the title's own anchor
rather than the `_top` one above it.

## Retrieval

Retrieval augmented generation combines a retriever with a generator so that
answers cite their sources instead of asserting them. The retriever decides
what the generator is allowed to know.

![a chart of recall against corpus size]({base}admin/fixtures/chart.png)

## Fusion

Reciprocal rank fusion merges ranked lists from incomparable scorers by rank
alone, which is why two branches can disagree entirely on score scale and
still fuse into one ordering.

**Why rank and not score**

Because a CLIP cosine near 0.3 and a bge cosine near 0.7 can describe equally
strong matches, and normalising them against each other assumes a calibration
neither model promises.

## Retrieval

A second section with the same heading as an earlier one, which must receive
the `-1` suffix while the first keeps the bare anchor — the numbering GitHub
and Substack both produce.
"""


@router.get("/fixtures/tiny.md")
def fixture_markdown(request: Request) -> Response:
    """A tiny markdown post — the post lifecycle contract test registers THIS
    URL so a real end-to-end ingest needs no external network.

    Unauthenticated for the same reason as tiny.pdf: the worker's fetch sends
    no bearer token, and the content is fixture text.
    """
    return Response(content=_FIXTURE_POST.format(base=request.base_url),
                    media_type="text/markdown")


def _fixture_png(width: int, height: int) -> bytes:
    """A plain chart-like raster: light ground, gridlines, a rising series.

    Generated rather than committed — no binary fixtures in git, and the shape
    is the point. 1200x100 is what a Substack section divider looks like to
    the heuristic layer; 600x400 is what a figure looks like.
    """
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (252, 252, 250))
    draw = ImageDraw.Draw(img)
    step = max(8, width // 12)
    for x in range(0, width, step):
        draw.line([(x, 0), (x, height)], fill=(216, 216, 216))
    for y in range(0, height, max(8, height // 8)):
        draw.line([(0, y), (width, y)], fill=(216, 216, 216))
    draw.line([(0, height - 1), (width, height - 1)], fill=(40, 40, 40), width=2)
    draw.line([(1, 0), (1, height)], fill=(40, 40, 40), width=2)
    series = [(x, height - 4 - int((height - 12) * (x / max(1, width)) ** 0.6))
              for x in range(0, width, max(2, width // 60))]
    draw.line(series, fill=(30, 90, 200), width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@router.get("/fixtures/banner.png")
def fixture_banner() -> Response:
    """1200x100 — dropped by the aspect-ratio rule before the classifier ever
    sees it, which is the deterministic half of the image gate."""
    return Response(content=_fixture_png(1200, 100), media_type="image/png")


@router.get("/fixtures/chart.png")
def fixture_chart() -> Response:
    """600x400 — clears every heuristic, so its fate is the classifier's."""
    return Response(content=_fixture_png(600, 400), media_type="image/png")
