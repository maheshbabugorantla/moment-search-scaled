"""Per-paper ingest pipeline — a Prefect flow mirroring the video one.

pending -> queued -> fetching -> parsing -> chunking -> embedding
        -> indexed | skipped | failed

Stages:
  1. fetch   stream the PDF into worker scratch + a content-addressed object
             (paper.fetch_paper); duplicate check via content hash
  2. parse   per-page text extraction (paper.parse_pdf)
  3. chunk   page-aware semantic chunks (rag/chunk.py)
  4. embed   bge text embeddings, batched -> idempotent Qdrant upsert into the
             TEXT collection (moments_text) — the same collection the video
             transcript branch writes, because paper chunks are text and the
             CLIP collection is 512-d image space. `kind: "paper"` and `page`
             ride on every payload.

Crash-safe ordering, built in rather than retrofitted in Epic 5: the row
becomes `indexed` ONLY after the Qdrant upsert returns, and point ids are
deterministic (doc_id + chunk idx), so a redelivered run overwrites instead of
duplicating. Retries live on the network stages (fetch, embed); parse errors
are deterministic and fail immediately — and PaperSourceError short-circuits
the fetch retries too (a 404 stays a 404 however often you ask).

Postgres remains the business-status source of truth, exactly as for video.
"""
from __future__ import annotations

from pathlib import Path

from prefect import flow, task

from .. import db
from ..config import (PAPER_CHUNK_CHARS, PAPER_CHUNK_OVERLAP,
                      PAPER_EMBED_BATCH, TEXT_EMBED_VERSION)
from ..rag import vector_store
from ..rag.chunk import chunk_pages
from ..rag.embeddings import embed_docs
from .paper import fetch_paper, parse_pdf, retry_unless_source_error


@task(name="fetch-paper", retries=2, retry_delay_seconds=[30, 120],
      retry_condition_fn=retry_unless_source_error)
def t_fetch_paper(doc_id: str, user_id: str) -> dict:
    """URI -> scratch file + durable object; duplicate check via content hash.

    Returns {} when the content duplicates an already-indexed source for this
    user (row marked 'skipped' — a plain outcome, not a retryable error).
    """
    db.set_status(doc_id, "fetching")
    row = db.get_video(doc_id)
    if row is None:
        raise ValueError(f"no manifest row for {doc_id}")
    handle = fetch_paper(row["uri"], user_id, doc_id)
    db.set_status(doc_id, "fetching", source_hash=handle["content_hash"],
                  storage_key=handle["storage_key"])

    dup = db.find_duplicate(user_id, handle["content_hash"], exclude_id=doc_id)
    if dup:
        Path(handle["scratch_path"]).unlink(missing_ok=True)
        db.set_status(doc_id, "skipped", error=f"duplicate of {dup['id']}")
        return {}
    return handle


@task(name="parse-paper")  # no retries — a parse failure is deterministic
def t_parse(doc_id: str, path: str) -> list[str]:
    db.set_status(doc_id, "parsing")
    pages = parse_pdf(Path(path))
    if not any(p.strip() for p in pages):
        # Every page scanned/image-only. Indexing nothing while reporting
        # `indexed` would be a lie; OCR is a corpus decision, not a default.
        raise RuntimeError(
            f"no extractable text on any of the {len(pages)} pages — "
            "a scanned PDF needs OCR, which this pipeline does not do")
    return pages


@task(name="chunk-paper")  # no retries — pure function of the parse output
def t_chunk(doc_id: str, pages: list[str]) -> list[dict]:
    db.set_status(doc_id, "chunking")
    chunks = chunk_pages(pages, max_chars=PAPER_CHUNK_CHARS,
                         overlap_chars=PAPER_CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("chunking produced nothing despite extractable text")
    print(f"[chunk] {doc_id}: {len(pages)} pages -> {len(chunks)} chunks")
    return [{"page": c.page, "text": c.text} for c in chunks]


@task(name="embed-upsert-paper", retries=2, retry_delay_seconds=60)
def t_embed_upsert(doc_id: str, user_id: str, chunks: list[dict]) -> int:
    """Batched bge embeddings -> idempotent upsert -> `indexed`.

    The status write comes strictly AFTER the last upsert returns — marking
    first and upserting second is the exact bug the Epic 5 resilience gate
    hunts for.
    """
    db.set_status(doc_id, "embedding", progress=0.0)
    vector_store.ensure_text_collection()
    vector_store.delete_video(user_id, doc_id)  # drop stale points from prior runs

    total = 0
    for start in range(0, len(chunks), PAPER_EMBED_BATCH):
        batch = chunks[start:start + PAPER_EMBED_BATCH]
        vecs = embed_docs([c["text"] for c in batch])
        vector_store.upsert_chunks(
            user_id, doc_id, vecs,
            payloads=[{"user_id": user_id, "video_id": doc_id,
                       "kind": "paper", "modality": "text", "page": c["page"],
                       "text": c["text"], "embed_version": TEXT_EMBED_VERSION}
                      for c in batch],
            start_idx=start,  # ids stay unique across batches
        )
        total += len(batch)
        db.set_progress(doc_id, total / len(chunks))
    # frame_count doubles as "how many units were indexed" — chunks, for a paper.
    db.set_status(doc_id, "indexed", frame_count=total,
                  embed_version=TEXT_EMBED_VERSION, progress=1.0)
    return total


@flow(name="ms-ingest-paper", log_prints=True, timeout_seconds=1800)
def ingest_paper(doc_id: str, user_id: str) -> dict:
    attempt = db.bump_attempts(doc_id)
    scratch: str | None = None
    try:
        handle = t_fetch_paper(doc_id, user_id)
        if not handle:  # duplicate — already marked 'skipped' by t_fetch_paper
            print(f"[ingest-paper] {doc_id} skipped (duplicate content)")
            return {"doc_id": doc_id, "skipped": True}
        scratch = handle["scratch_path"]
        pages = t_parse(doc_id, scratch)
        chunks = t_chunk(doc_id, pages)
        n = t_embed_upsert(doc_id, user_id, chunks)
        print(f"[ingest-paper] {doc_id} indexed: {handle['page_count']} pages, "
              f"{n} chunks (attempt {attempt})")
        return {"doc_id": doc_id, "pages": handle["page_count"], "chunks": n}
    except Exception as exc:
        db.set_status(doc_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise  # Prefect marks the run Failed; full trace in the Cloud UI
    finally:
        if scratch:  # scratch only — the durable copy is the content-hash object
            Path(scratch).unlink(missing_ok=True)
