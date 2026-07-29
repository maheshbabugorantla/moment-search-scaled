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

from fastapi import APIRouter, Depends, HTTPException, Query
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

    Reports; it does not measure. `pct` renders whatever the column holds, and
    nothing writes it yet (REC-310 does) — so an `indexed` source currently
    reports `pct: 0`. Treat `status` as the completion signal, not `pct`, until
    that lands.
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
