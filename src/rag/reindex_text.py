"""Rebuild the text collection after an embedding-model change.

Changing TEXT_EMBED_MODEL changes the vector dimension, and Qdrant collections
are fixed-width — so every text point in the corpus has to be re-embedded. This
is the operator tool `vector_store.DimensionMismatch` points at.

    python -m src.rag.reindex_text --report          # what is indexed now
    python -m src.rag.reindex_text --drop            # drop, then rebuild all
    python -m src.rag.reindex_text --videos          # transcripts only

Two rebuild paths, for two different reasons:

* **Videos are re-embedded IN PROCESS, from captions.** A video's transcript
  needs only its URL — not the media file, not the sampled frames, which live
  in the CLIP collection and are untouched by a text-model change. Re-running
  the whole video flow would re-download and re-sample 31 videos for nothing,
  turning a half-hour job into a day and putting the corpus at the mercy of
  YouTube's bot detection.
* **Papers and posts go back through their real flows**, by being marked
  pending for the dispatcher. Their fetch is cheap (content-addressed objects,
  local fixtures) and their chunking is the part that must not drift: a
  reimplementation here would be a second chunker, and two chunkers disagreeing
  means point ids that no longer overwrite. The flows are the tested path.

Failure is loud on purpose. t_transcript swallows every exception by design —
a video with no captions must not fail its ingest — but during a rebuild that
same silence turns a half-empty index into a green run, so this counts chunks
per source before and after and reports every source that lost them.
"""
from __future__ import annotations

import argparse
import sys
import time

from qdrant_client.http import models as qm

from .. import db
from ..config import TEXT_COLLECTION, TEXT_EMBED_VERSION
from . import vector_store


def _chunk_counts() -> dict[str, int]:
    """text points per source id, straight from Qdrant."""
    c = vector_store.client()
    if not c.collection_exists(TEXT_COLLECTION):
        return {}
    out: dict[str, int] = {}
    offset = None
    while True:
        points, offset = c.scroll(collection_name=TEXT_COLLECTION, limit=1000,
                                  offset=offset, with_payload=["video_id"])
        for p in points:
            vid = (p.payload or {}).get("video_id")
            if vid:
                out[vid] = out.get(vid, 0) + 1
        if offset is None:
            return out


def _sources() -> list[dict]:
    with db.pool().connection() as conn:
        return conn.execute(
            "SELECT id, user_id, kind, uri, url, title, status FROM ms_videos "
            "WHERE status = 'indexed' ORDER BY kind, id").fetchall()


def report() -> None:
    counts = _chunk_counts()
    rows = _sources()
    by_kind: dict[str, list[int]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(counts.get(r["id"], 0))
    print(f"collection: {TEXT_COLLECTION}   embed_version: {TEXT_EMBED_VERSION}")
    for kind, vals in sorted(by_kind.items()):
        empty = sum(1 for v in vals if v == 0)
        flag = f"   <-- {empty} with ZERO text points" if empty else ""
        print(f"  {kind:6s} sources={len(vals):3d}  text points={sum(vals):6d}{flag}")
    for r in rows:
        if counts.get(r["id"], 0) == 0 and r["kind"] != "video":
            print(f"    empty: {r['id']} {(r['title'] or r['uri'])[:60]}")


def drop() -> None:
    c = vector_store.client()
    if c.collection_exists(TEXT_COLLECTION):
        c.delete_collection(TEXT_COLLECTION)
        print(f"[drop] deleted {TEXT_COLLECTION}")
    else:
        print(f"[drop] {TEXT_COLLECTION} did not exist")
    # Recreate immediately at the CONFIGURED dimension, so a wrong env var
    # fails here — with nothing indexed yet — rather than after a 30-minute
    # rebuild has been fed into a collection of the wrong width.
    vector_store.ensure_text_collection()
    info = c.get_collection(TEXT_COLLECTION)
    print(f"[drop] recreated at {info.config.params.vectors.size}-d")


def reindex_videos(before: dict[str, int]) -> list[str]:
    """Re-embed every YouTube transcript in process, from captions only."""
    from ..ingest.transcript import chunk_cues, fetch_transcript
    from .embeddings import embed_docs

    lost: list[str] = []
    rows = [r for r in _sources() if r["kind"] == "video" and r.get("url")]
    for i, r in enumerate(rows, 1):
        vid, uid = r["id"], r["user_id"]
        had = before.get(vid, 0)
        # One retry. Qdrant dropped a connection mid-rebuild on the first run
        # ("Server disconnected without sending a response"), which cost a
        # video its entire transcript for a blip that a second attempt clears.
        # A rebuild is long enough that a transient failure is expected rather
        # than exceptional, and re-running the whole job to recover one video
        # is the wrong unit of retry.
        for attempt in (1, 2):
            try:
                chunks = chunk_cues(fetch_transcript(r["url"], vid))
                if not chunks:
                    raise RuntimeError("no captions returned")
                vector_store.ensure_text_collection()
                vecs = embed_docs([c["text"] for c in chunks])
                vector_store.upsert_chunks(uid, vid, vecs, payloads=[
                    {"user_id": uid, "video_id": vid, "modality": "text",
                     "t_start": c["t_start"], "t_end": c["t_end"],
                     "ms": int(c["t_start"] * 1000), "text": c["text"],
                     "embed_version": TEXT_EMBED_VERSION} for c in chunks])
                note = " (retry)" if attempt == 2 else ""
                print(f"[{i:2d}/{len(rows)}] {vid} {len(chunks):4d} chunks "
                      f"(was {had}){note} {(r['title'] or '')[:44]}")
                break
            except Exception as exc:
                if attempt == 1:
                    print(f"[{i:2d}/{len(rows)}] {vid} retrying after "
                          f"{type(exc).__name__}")
                    time.sleep(5)
                    continue
                # Loud, unlike t_transcript: during a rebuild a swallowed
                # failure is a silently half-empty index reporting success.
                lost.append(vid)
                print(f"[{i:2d}/{len(rows)}] {vid} FAILED {type(exc).__name__}: "
                      f"{exc} (had {had} chunks)")
    return lost


def requeue_documents() -> int:
    """Mark papers and posts pending so the dispatcher re-runs their flows."""
    with db.pool().connection() as conn:
        rows = conn.execute(
            "UPDATE ms_videos SET status = 'pending', error = NULL, pct = 0, "
            "stage = NULL, progress = NULL, updated_at = now() "
            "WHERE kind IN ('paper', 'post') AND status = 'indexed' "
            "RETURNING id").fetchall()
    print(f"[requeue] {len(rows)} document(s) marked pending for the dispatcher")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true", help="show counts and exit")
    ap.add_argument("--drop", action="store_true",
                    help="drop and recreate the collection, then rebuild everything")
    ap.add_argument("--videos", action="store_true",
                    help="re-embed video transcripts only")
    args = ap.parse_args()

    if args.report or not (args.drop or args.videos):
        report()
        return 0

    before = _chunk_counts()
    print(f"[before] {sum(before.values())} text points across "
          f"{len(before)} sources\n")

    if args.drop:
        drop()
        # BEFORE the video loop, not after. On the first run this sat behind
        # ten minutes of transcript work, the connection pool timed out, and
        # papers and posts were left marked `indexed` with zero vectors — the
        # manifest asserting something the index could not back. Requeueing
        # first also lets the dispatcher rebuild documents in parallel.
        requeue_documents()
    lost = reindex_videos(before)

    print()
    if lost:
        print(f"!! {len(lost)} video(s) lost their transcript: {', '.join(lost)}")
        print("   These are searchable by frame only until re-run.")
    report()
    return 1 if lost else 0


if __name__ == "__main__":
    sys.exit(main())
