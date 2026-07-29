"""Acquiring a paper (PDF) into object storage, idempotent by content hash.

The first worker-side stage of document ingest: pull the PDF the admin
registered and put it somewhere durable, so nothing downstream re-downloads it
and a crashed run can resume from local state. Only moves bytes — parsing
lives in parse_pdf / rag.chunk.

Two URI schemes, one contract — a scratch file plus a durable object:
  * https:// (and http://) — streamed down in 1 MB chunks, hashed as it lands,
    never fully buffered in memory.
  * storage://<key>        — an object already in our bucket (uploaded via a
    presigned PUT); streamed to scratch from there.

The durable object is keyed by content hash — papers/{user_id}/{sha256}.pdf —
so the same paper registered twice (same or different URI) resolves to ONE
object and the second run skips the upload entirely.

Failure classification matters here: Prefect retries the fetch task, and a 404
or a not-a-PDF is deterministic — retrying it three times over 150s would just
triple the time-to-failed. Those raise PaperSourceError, which the task's
retry_condition_fn refuses to retry; timeouts, connection resets and 5xx raise
plain exceptions and get the retry policy.

Debug entrypoint (REC-307's verify):
    python -m src.ingest.paper <uri> [user_id]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .. import storage
from ..config import MAX_PAPER_MB, PAPER_KEY_PREFIX
from .download import stream_to_file
from .errors import SourceError, retry_unless_source_error  # noqa: F401
from .fetch import scratch_dir, sha256_file


# The classification moved to ingest/errors.py once the video flow needed it
# too. The old name stays as an alias: it reads correctly at the paper call
# sites, and nothing importing it has to change.
PaperSourceError = SourceError


def paper_key(user_id: str, content_hash: str) -> str:
    """Durable object key — user-scoped like every other prefix (tenant
    isolation at the path level), content-addressed within the tenant."""
    return f"{PAPER_KEY_PREFIX}{user_id}/{content_hash}.pdf"


def check_pdf_magic(first_bytes: bytes) -> None:
    """A PDF starts with %PDF- (possibly after a UTF-8 BOM or stray junk some
    generators emit — the spec allows the header within the first 1024 bytes)."""
    if b"%PDF-" not in first_bytes[:1024]:
        raise PaperSourceError(
            "not a PDF: the content does not start with a %PDF header "
            "(got something else — an HTML error page, most likely)")


def _check_size(byte_count: int) -> None:
    if byte_count > MAX_PAPER_MB * (1 << 20):
        raise PaperSourceError(
            f"PDF exceeds the {MAX_PAPER_MB} MB limit "
            f"({byte_count / (1 << 20):.0f} MB so far)")


def _stream_http(uri: str, dest: Path) -> tuple[str, int]:
    """GET the URI to `dest`, hashing and cap-checking as chunks arrive.

    The body moved to ingest/download.py once the post flow needed the same
    guards and the same retryable-vs-not classification. MAX_PAPER_MB is read
    HERE, at call time, so the cap stays a property of this module.
    """
    return stream_to_file(uri, dest, max_mb=MAX_PAPER_MB, what="PDF",
                          first_chunk_check=check_pdf_magic)


def _page_count(path: Path) -> int:
    import pymupdf

    try:
        with pymupdf.open(path) as doc:
            n = doc.page_count
    except Exception as exc:
        raise PaperSourceError(f"unreadable PDF: {exc}") from exc
    if n < 1:
        raise PaperSourceError("the PDF has no pages")
    return n


def parse_pdf(path: Path) -> list[str]:
    """Stored bytes -> one NORMALIZED text string per page (index 0 = page 1).

    The list length always equals the page count — image-only pages yield an
    empty string rather than being dropped, so downstream chunking can skip
    them WITHOUT losing the true page numbering (rag/chunk.py relies on this).
    A PDF that opens but yields no text anywhere is the caller's call to fail —
    that's a corpus decision (OCR or reject), not a parse error.

    Pages go through textnorm.normalize_page before anything sees them:
    pymupdf returns what the page LOOKS like, which includes ligature glyphs
    and typesetter hyphenation that are not part of the words. Normalizing
    here rather than at chunk time means chunk boundaries — and the point ids
    derived from them — are computed from clean text.
    """
    import pymupdf

    from .textnorm import normalize_page, typography_report

    try:
        with pymupdf.open(path) as doc:
            raw = [page.get_text("text") for page in doc]
    except Exception as exc:
        raise PaperSourceError(f"unreadable PDF: {exc}") from exc

    # Log what typography this source carried. REC-338 was found by an ad-hoc
    # script a month after the corpus was indexed; a one-line summary per
    # ingest makes "how mangled is this PDF" answerable at the time.
    found: Counter[str] = Counter()
    for page_text in raw:
        found.update(typography_report(page_text))
    if found:
        print(f"[parse] {path.name}: normalized " +
              ", ".join(f"{v} {k}" for k, v in sorted(found.items()) if v))
    return [normalize_page(page_text) for page_text in raw]


def fetch_paper(uri: str, user_id: str, doc_id: str) -> dict:
    """URI -> scratch file + durable content-addressed object.

    Returns {"storage_key", "content_hash", "byte_size", "page_count",
    "scratch_path"}. The scratch file is the caller's to delete (the flow's
    `finally`, mirroring the video pipeline); the durable object stays.
    """
    dest = scratch_dir() / f"{doc_id}.pdf"
    if uri.startswith("storage://"):
        storage.download_to(uri[len("storage://"):], dest)
        with dest.open("rb") as fh:
            check_pdf_magic(fh.read(1024))
        _check_size(dest.stat().st_size)
        content_hash, size = sha256_file(dest), dest.stat().st_size
    else:
        content_hash, size = _stream_http(uri, dest)

    pages = _page_count(dest)
    key = paper_key(user_id, content_hash)
    if not storage.exists(key):  # same content already stored -> reuse, no re-upload
        storage.upload_file(dest, key, content_type="application/pdf")
    return {"storage_key": key, "content_hash": content_hash,
            "byte_size": size, "page_count": pages, "scratch_path": str(dest)}


if __name__ == "__main__":  # python -m src.ingest.paper <uri> [user_id]
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.ingest.paper <uri> [user_id]")
    _uri = sys.argv[1]
    _user = sys.argv[2] if len(sys.argv) > 2 else "default"
    handle = fetch_paper(_uri, _user, "doc_debug")
    Path(handle.pop("scratch_path")).unlink(missing_ok=True)
    print(json.dumps(handle, indent=2))
