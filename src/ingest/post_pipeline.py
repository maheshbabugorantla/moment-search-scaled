"""Per-post ingest pipeline — a Prefect flow mirroring the paper one.

pending -> queued -> fetching -> parsing -> chunking -> embedding
        -> indexed | skipped | failed

Stages:
  1. fetch   stream the markdown into worker scratch + a content-addressed
             object (post.fetch_post); duplicate check via content hash
  2. parse   text -> sections carrying anchor, heading path and image refs
             (post.read_markdown + post.parse_markdown)
  3. chunk   section-bounded chunks (rag/chunk.py chunk_markdown)
  4. embed   bge text embeddings, batched -> idempotent Qdrant upsert into the
             TEXT collection (moments_text) — the same collection papers and
             video transcripts use. `kind: "post"` and the heading `anchor`
             ride on every payload.

No new status and no new stage vocabulary: a post walks the document
lifecycle a paper already defined. What differs is one field on the payload —
`anchor` instead of `page` — which is the whole point of the kind and is
exactly the kind of difference that should NOT need its own state machine.

Crash-safe ordering, as everywhere else: the row becomes `indexed` ONLY after
the Qdrant upsert returns, and point ids are deterministic (doc_id + chunk
idx), so a redelivered run overwrites instead of duplicating.
"""
from __future__ import annotations

from pathlib import Path

from prefect import flow, task

from .. import db
from ..config import (POST_CHUNK_CHARS, POST_CHUNK_OVERLAP, POST_EMBED_BATCH,
                      TEXT_EMBED_VERSION)
from ..rag import vector_store
from ..rag.chunk import chunk_markdown
from ..rag.embeddings import embed_docs
from .errors import SourceError, retry_unless_source_error
from .post import fetch_post, parse_markdown, read_markdown


class _Section:
    """The four fields chunk_markdown reads, rebuilt from the serialised parse
    result. Deliberately not ingest.post.Section: what crosses a task boundary
    is data, so the chunk stage reconstructs what it needs rather than
    depending on the parse module's dataclass shape."""

    __slots__ = ("anchor", "heading", "anchor_native", "paragraphs")

    def __init__(self, anchor: str, heading: str, anchor_native: bool,
                 paragraphs: list[str]):
        self.anchor = anchor
        self.heading = heading
        self.anchor_native = anchor_native
        self.paragraphs = paragraphs


@task(name="fetch-post", retries=2, retry_delay_seconds=[30, 120],
      retry_condition_fn=retry_unless_source_error)
def t_fetch_post(doc_id: str, user_id: str) -> dict:
    """URI -> scratch file + durable object; duplicate check via content hash.

    Returns {} when the content duplicates an already-indexed source for this
    user (row marked 'skipped' — a plain outcome, not a retryable error).
    """
    db.set_status(doc_id, "fetching")
    row = db.get_video(doc_id)
    if row is None:
        # Deleted between dispatch and pickup (test cleanup, user delete).
        # Deterministic — a row does not come back — so this skips the retry
        # ladder instead of holding a worker slot for 150s to fail identically.
        raise SourceError(f"no manifest row for {doc_id}")
    handle = fetch_post(row["uri"], user_id, doc_id)
    handle["title"] = row.get("title") or ""
    db.set_status(doc_id, "fetching", source_hash=handle["content_hash"],
                  storage_key=handle["storage_key"])

    dup = db.find_duplicate(user_id, handle["content_hash"], exclude_id=doc_id)
    if dup:
        Path(handle["scratch_path"]).unlink(missing_ok=True)
        db.set_status(doc_id, "skipped", error=f"duplicate of {dup['id']}")
        return {}
    db.set_stage(doc_id, stage="fetch", pct=20)
    return handle


@task(name="parse-post")  # no retries — a parse failure is deterministic
def t_parse_post(doc_id: str, path: str, title: str) -> list[dict]:
    db.set_status(doc_id, "parsing")
    sections = parse_markdown(read_markdown(Path(path)), title=title)
    if not any(s.paragraphs for s in sections):
        # Markdown that parsed but carries no prose: an image dump, or a link
        # list. Reporting `indexed` with nothing indexed would be a lie.
        raise RuntimeError(
            "no prose in the markdown — nothing to index "
            "(is this an image-only export?)")
    native = sum(1 for s in sections if s.anchor_native)
    images = sum(len(s.images) for s in sections)
    print(f"[parse] {doc_id}: {len(sections)} sections "
          f"({native} native anchors, {len(sections) - native} synthesised), "
          f"{images} image ref(s)")
    db.set_stage(doc_id, stage="parse", pct=35)
    # Prefect serialises task results; hand on plain dicts rather than the
    # frozen dataclasses so a resumed run deserialises without importing them.
    return [{"anchor": s.anchor, "heading": s.heading,
             "anchor_native": s.anchor_native,
             "paragraphs": list(s.paragraphs)} for s in sections]


@task(name="chunk-post")  # no retries — pure function of the parse output
def t_chunk_post(doc_id: str, sections: list[dict]) -> list[dict]:
    db.set_status(doc_id, "chunking")
    chunks = chunk_markdown([_Section(**s) for s in sections],
                            max_chars=POST_CHUNK_CHARS,
                            overlap_chars=POST_CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("chunking produced nothing despite parsed prose")
    print(f"[chunk] {doc_id}: {len(sections)} sections -> {len(chunks)} chunks")
    db.set_stage(doc_id, stage="chunk", pct=45)
    return [{"anchor": c.anchor, "heading": c.heading,
             "anchor_native": c.anchor_native, "text": c.text} for c in chunks]


@task(name="embed-upsert-post", retries=2, retry_delay_seconds=60)
def t_embed_upsert_post(doc_id: str, user_id: str, chunks: list[dict]) -> int:
    """Batched bge embeddings -> idempotent upsert -> `indexed`.

    The status write comes strictly AFTER the last upsert returns — marking
    first and upserting second is the exact bug the Epic 5 resilience gate
    hunts for.
    """
    db.set_status(doc_id, "embedding", progress=0.0)
    vector_store.ensure_text_collection()
    vector_store.delete_video(user_id, doc_id)  # drop stale points from prior runs

    total = 0
    for start in range(0, len(chunks), POST_EMBED_BATCH):
        batch = chunks[start:start + POST_EMBED_BATCH]
        vecs = embed_docs([c["text"] for c in batch])
        vector_store.upsert_chunks(
            user_id, doc_id, vecs,
            payloads=[{"user_id": user_id, "video_id": doc_id,
                       "kind": "post", "modality": "text",
                       "anchor": c["anchor"], "heading": c["heading"],
                       "anchor_native": c["anchor_native"],
                       "text": c["text"], "embed_version": TEXT_EMBED_VERSION}
                      for c in batch],
            start_idx=start,  # ids stay unique across batches
        )
        total += len(batch)
        db.set_progress(doc_id, total / len(chunks))
        # 45 -> 95 across the embed batches; the last 5 points arrive with
        # `indexed` so pct hits 100 exactly when the status does.
        db.set_stage(doc_id, pct=45 + int(50 * total / len(chunks)))
    # frame_count doubles as "how many units were indexed" — chunks, for a post.
    db.set_status(doc_id, "indexed", frame_count=total,
                  embed_version=TEXT_EMBED_VERSION, progress=1.0)
    db.set_stage(doc_id, stage="embed", pct=100)
    return total


@flow(name="ms-ingest-post", log_prints=True, timeout_seconds=1800)
def ingest_post(doc_id: str, user_id: str) -> dict:
    attempt = db.bump_attempts(doc_id)
    scratch: str | None = None
    try:
        handle = t_fetch_post(doc_id, user_id)
        if not handle:  # duplicate — already marked 'skipped' by t_fetch_post
            print(f"[ingest-post] {doc_id} skipped (duplicate content)")
            return {"doc_id": doc_id, "skipped": True}
        scratch = handle["scratch_path"]
        sections = t_parse_post(doc_id, scratch, handle["title"])
        chunks = t_chunk_post(doc_id, sections)
        n = t_embed_upsert_post(doc_id, user_id, chunks)
        print(f"[ingest-post] {doc_id} indexed: {len(sections)} sections, "
              f"{n} chunks (attempt {attempt})")
        return {"doc_id": doc_id, "sections": len(sections), "chunks": n}
    except Exception as exc:
        db.set_status(doc_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise  # Prefect marks the run Failed; full trace in the Cloud UI
    finally:
        if scratch:  # scratch only — the durable copy is the content-hash object
            Path(scratch).unlink(missing_ok=True)
