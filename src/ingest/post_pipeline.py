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
                      POST_INDEX_IMAGES, TEXT_EMBED_VERSION)
from ..rag import vector_store
from ..rag.chunk import chunk_markdown
from ..rag.embeddings import embed_docs
from .errors import SourceError, retry_unless_source_error
from .post import fetch_post, parse_markdown, read_markdown, split_frontmatter


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
    handle["uri"] = row["uri"]  # relative image refs resolve against it
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
def t_parse_post(doc_id: str, path: str, title: str) -> dict:
    db.set_status(doc_id, "parsing")
    meta, body = split_frontmatter(read_markdown(Path(path)))
    # An exporter's front matter knows the canonical post URL; the registered
    # URI may be a storage:// key no reader can follow. Prefer the former for
    # the deeplink, and adopt the exported title when the operator gave none.
    title = title or meta.get("title", "")
    sections = parse_markdown(body, title=title)
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
    return {
        "title": title,
        "source_url": meta.get("url", ""),
        "author": meta.get("author", ""),
        "sections": [{"anchor": s.anchor, "heading": s.heading,
                      "anchor_native": s.anchor_native,
                      "paragraphs": list(s.paragraphs),
                      "images": [{"url": i.url, "alt": i.alt,
                                  "hero": i.before_first_heading and i.position == 0}
                                 for i in s.images]} for s in sections],
    }


@task(name="images-post")  # best-effort by contract — see the flow's call site
def t_images_post(doc_id: str, user_id: str, sections: list[dict],
                  base_uri: str = "") -> int:
    """Worthiness-gate the post's images and index the survivors.

    Kept images go into the CLIP collection through the SAME frame_key layout
    video thumbnails use, so every existing thumbnail and citation-rendering
    path works on a post without knowing posts exist.

    Never raises. An image is an enhancement; a post whose chart failed to
    download is still a correctly indexed post, and this stage carries no
    retries. That is the transcript branch's contract, for the same reason.
    """
    refs = [(s["anchor"], img) for s in sections for img in s.get("images", ())]
    if not POST_INDEX_IMAGES or not refs:
        return 0
    try:
        import numpy as np

        from .. import storage
        from ..config import EMBED_VERSION
        from . import post_images

        prompts = post_images.prompt_bank()
        kept: list[tuple[int, str, bytes, post_images.Verdict]] = []
        for idx, (anchor, img) in enumerate(refs):
            url = post_images.resolve(img["url"], base_uri)
            verdict = post_images.judge(url, alt=img.get("alt", ""),
                                        hero=bool(img.get("hero")),
                                        prompts=prompts)
            mark = "keep" if verdict.keep else "drop"
            print(f"[images] {doc_id} #{anchor} {mark}: {verdict.reason} "
                  f"— {url[:80]}")
            if verdict.keep:
                kept.append((idx, anchor, verdict.jpeg, verdict))
        if not kept:
            return 0

        storage.delete_prefix(storage.frame_prefix(user_id, doc_id))
        for idx, anchor, jpeg, _ in kept:
            storage.put_bytes(storage.frame_key(user_id, doc_id, idx), jpeg,
                              "image/jpeg")
        vector_store.ensure_collection()
        # Reuse the vectors the classifier already computed — the alternative
        # is a second CLIP pass over the same bytes for no new information.
        vectors = np.stack([v.vector for _, _, _, v in kept])
        vector_store.upsert_frames(
            user_id, doc_id,
            ids=[idx for idx, _, _, _ in kept],
            vectors=vectors,
            payloads=[{"user_id": user_id, "video_id": doc_id, "kind": "post",
                       "modality": "frame", "anchor": anchor, "idx": idx,
                       "img_class": v.img_class, "img_score": v.img_score,
                       "embed_version": EMBED_VERSION}
                      for idx, anchor, _, v in kept],
        )
        print(f"[images] {doc_id}: {len(kept)}/{len(refs)} image(s) indexed")
        return len(kept)
    except Exception as exc:
        print(f"[images] {doc_id}: failed ({type(exc).__name__}: {exc}) "
              "— text-only")
        return 0


@task(name="chunk-post")  # no retries — pure function of the parse output
def t_chunk_post(doc_id: str, sections: list[dict]) -> list[dict]:
    db.set_status(doc_id, "chunking")
    chunks = chunk_markdown(
        [_Section(s["anchor"], s["heading"], s["anchor_native"],
                  s["paragraphs"]) for s in sections],
        max_chars=POST_CHUNK_CHARS, overlap_chars=POST_CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("chunking produced nothing despite parsed prose")
    print(f"[chunk] {doc_id}: {len(sections)} sections -> {len(chunks)} chunks")
    db.set_stage(doc_id, stage="chunk", pct=45)
    return [{"anchor": c.anchor, "heading": c.heading,
             "anchor_native": c.anchor_native, "text": c.text} for c in chunks]


@task(name="embed-upsert-post", retries=2, retry_delay_seconds=60)
def t_embed_upsert_post(doc_id: str, user_id: str, chunks: list[dict],
                        source_url: str = "") -> int:
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
                       # The deeplink Epic 4 will render is source_url#anchor.
                       # Stored per point so a citation needs no second read.
                       "source_url": source_url,
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
        parsed = t_parse_post(doc_id, scratch, handle["title"])
        sections = parsed["sections"]
        chunks = t_chunk_post(doc_id, sections)
        n = t_embed_upsert_post(doc_id, user_id, chunks,
                                parsed["source_url"] or handle["uri"])
        # Images AFTER the text upsert, not inside parse where they logically
        # belong: t_embed_upsert_post opens with delete_video(), which purges
        # BOTH collections to clear a previous run's points. Classifying first
        # would index images and then delete them. The video flow orders its
        # transcript branch after embed-index for exactly this reason.
        img = t_images_post(doc_id, user_id, sections, handle["uri"])
        print(f"[ingest-post] {doc_id} indexed: {len(sections)} sections, "
              f"{n} chunks + {img} image(s) (attempt {attempt})")
        return {"doc_id": doc_id, "sections": len(sections), "chunks": n,
                "images": img}
    except Exception as exc:
        db.set_status(doc_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise  # Prefect marks the run Failed; full trace in the Cloud UI
    finally:
        if scratch:  # scratch only — the durable copy is the content-hash object
            Path(scratch).unlink(missing_ok=True)
