"""Qdrant — one shared multi-tenant collection, one point per kept frame.

Multi-tenancy: every point carries user_id; the field has a tenant payload
index and every search / upsert / delete is user_id-filtered. NOT
collection-per-user (collection explosion); a huge tenant can graduate to a
dedicated collection later.

Memory profile (the frame-scale levers, all env flags, default ON):
  QDRANT_ON_DISK        original float vectors live on disk
  QDRANT_QUANTIZATION   int8 copies pinned in RAM (~4x smaller) do the search;
                        queries rescore the top candidates from the originals
  QDRANT_HNSW_ON_DISK   the HNSW graph lives on disk too

Point IDs are uuid5 of "{video_id}:{frame_idx}" — deterministic, so re-runs
overwrite instead of duplicating. Payloads are trimmed to filter/display
fields (user_id, video_id, ms, idx, embed_version); titles and URLs live in
Postgres and are joined at answer time.
"""
from __future__ import annotations

import uuid
from typing import Any, Iterable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..config import (
    CLIP_DIM,
    CLIP_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_HNSW_ON_DISK,
    QDRANT_LOCAL_PATH,
    QDRANT_ON_DISK,
    QDRANT_QUANTIZATION,
    QDRANT_URL,
    TEXT_COLLECTION,
    TEXT_EMBED_DIM,
)

_client: QdrantClient | None = None

# Dimensions of the stock sentence-transformers CLIP checkpoints — lets the
# API create the collection at boot without pulling in torch or downloading
# the model. Unknown/custom models: set CLIP_DIM, or the model gets loaded.
_KNOWN_DIMS = {
    "clip-ViT-B-32": 512,
    "clip-ViT-B-16": 512,
    "clip-ViT-L-14": 768,
    "clip-ViT-L-14-336": 768,
}


def _dim() -> int:
    if CLIP_DIM:
        return CLIP_DIM
    if CLIP_MODEL in _KNOWN_DIMS:
        return _KNOWN_DIMS[CLIP_MODEL]
    from .embeddings import embedding_dim  # last resort — loads the model

    return embedding_dim()


def client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None,
                                   timeout=60)
        else:  # embedded local instance — dev only, single-process
            _client = QdrantClient(path=QDRANT_LOCAL_PATH)
    return _client


def point_id(video_id: str, frame_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}:{frame_idx}"))


def _user_filter(user_id: str, video_id: str | None = None,
                 video_ids: list[str] | None = None) -> qm.Filter:
    must: list[qm.FieldCondition] = [
        qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))]
    if video_id:  # single-video scope (kept for /transcript-style calls)
        must.append(qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id)))
    elif video_ids:  # multi-select scope — query only the chosen videos
        must.append(qm.FieldCondition(key="video_id", match=qm.MatchAny(any=video_ids)))
    return qm.Filter(must=must)


class DimensionMismatch(RuntimeError):
    """The live collection's vector size disagrees with the configured model."""


def _check_dim(collection: str, dim: int) -> None:
    """Refuse to run against a collection built for a different model.

    Changing TEXT_EMBED_MODEL changes TEXT_EMBED_DIM (bge is 384,
    text-embedding-3-small 1536). `_ensure` only CREATES when the collection is
    absent, so on a dimension change it silently leaves the old one in place
    and every upsert fails deep inside the Qdrant client with a message about
    vector sizes — which reads like an embedding bug, not a schema one, and
    sends you looking in the wrong module.

    Deliberately raises rather than dropping. A dimension mismatch is
    ambiguous: it means either "I meant to change models" or "I typo'd an env
    var", and one of those readings costs the entire index. Rebuilding is an
    explicit operator action, and the message says exactly what to run.
    """
    c = client()
    try:
        info = c.get_collection(collection)
        cfg = info.config.params.vectors
        live = cfg.size if hasattr(cfg, "size") else None
    except Exception:
        return  # unreadable or absent — _ensure handles creation
    if live is not None and live != dim:
        raise DimensionMismatch(
            f"{collection} holds {live}-d vectors but the configured embedding "
            f"model wants {dim}-d. Nothing has been changed. Either restore the "
            f"previous TEXT_EMBED_MODEL / TEXT_EMBED_DIM, or rebuild the "
            f"collection and re-index every source:\n"
            f"    python -m src.rag.reindex_text --drop")


def _ensure(collection: str, dim: int) -> None:
    """Create a collection (low-RAM profile) + tenant/video payload indexes."""
    c = client()
    _check_dim(collection, dim)
    if not c.collection_exists(collection):
        c.create_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(
                size=dim,
                distance=qm.Distance.COSINE,
                on_disk=QDRANT_ON_DISK,
            ),
            hnsw_config=qm.HnswConfigDiff(on_disk=QDRANT_HNSW_ON_DISK),
            quantization_config=(
                qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(
                    type=qm.ScalarType.INT8, always_ram=True))
                if QDRANT_QUANTIZATION else None
            ),
        )
    # Tenant index on user_id: co-locates a tenant's points so per-user
    # searches touch a small slice of the index. video_id for delete/filter.
    try:
        c.create_payload_index(
            collection_name=collection, field_name="user_id",
            field_schema=qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD,
                                               is_tenant=True))
    except Exception:  # older server without is_tenant, or index already exists
        try:
            c.create_payload_index(collection_name=collection, field_name="user_id",
                                   field_schema=qm.PayloadSchemaType.KEYWORD)
        except Exception:
            pass
    try:
        c.create_payload_index(collection_name=collection, field_name="video_id",
                               field_schema=qm.PayloadSchemaType.KEYWORD)
    except Exception:
        pass
    # kind: filtered by search_text's paper/deck exclusion (and by Epic 4's
    # per-kind scoping later). Qdrant Cloud REJECTS filters on unindexed
    # fields ("Index required but not found"), so the index must exist before
    # the first filtered query, not after.
    try:
        c.create_payload_index(collection_name=collection, field_name="kind",
                               field_schema=qm.PayloadSchemaType.KEYWORD)
    except Exception:
        pass


def ensure_collection() -> None:
    """Visual (CLIP frame) collection."""
    _ensure(QDRANT_COLLECTION, _dim())


def ensure_text_collection() -> None:
    """Transcript (bge text) collection — the second branch."""
    _ensure(TEXT_COLLECTION, TEXT_EMBED_DIM)


def upsert_frames(user_id: str, video_id: str, ids: Iterable[int],
                  vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
    points = [
        qm.PointStruct(id=point_id(video_id, idx), vector=vec.tolist(), payload=payload)
        for idx, vec, payload in zip(ids, vectors, payloads)
    ]
    if points:
        client().upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)


def search(vector: np.ndarray, user_id: str, *, top_k: int,
           video_id: str | None = None,
           video_ids: list[str] | None = None) -> list[dict[str, Any]]:
    # Every kind, since Epic 4. Until then this branch excluded documents
    # because the citation builder read `fr["ms"]` unconditionally and a post
    # image — which lives HERE, reusing the frame_key layout deliberately, and
    # carries an `anchor` instead of an `ms` — 500ed the whole answer.
    #
    # What made that lift safe is not this line: it is that `_locator_payload`
    # now derives the locator from the winning hit per kind, so a hit without
    # an `ms` is a normal citation rather than a KeyError.
    flt = _user_filter(user_id, video_id, video_ids)
    try:
        hits = client().query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector.tolist(),
            limit=top_k,
            query_filter=flt,
            with_payload=True,
            search_params=qm.SearchParams(
                # Quantized search is lossy; rescore re-reads the full-precision
                # vectors from disk for the top candidates.
                quantization=qm.QuantizationSearchParams(rescore=True)
                if QDRANT_QUANTIZATION else None,
            ),
        ).points
    except Exception as exc:
        # Empty deployment (collection not created yet) is a "no results"
        # situation, not a 500 — the UI shows "no moments found".
        if "doesn't exist" in str(exc) or "Not found" in str(exc):
            return []
        raise
    return [{"score": float(h.score), **(h.payload or {})} for h in hits]


# ── Transcript (text) branch ─────────────────────────────────────────────────

def upsert_chunks(user_id: str, video_id: str, vectors: np.ndarray,
                  payloads: list[dict[str, Any]], start_idx: int = 0) -> None:
    """Text chunks into the text collection (transcripts AND paper chunks —
    both are text, one collection). IDs are uuid5 of '<video_id>:text:<i>' so
    re-runs overwrite, and never collide with frame ids. `start_idx` numbers a
    BATCHED upsert: without it every batch would restart at :text:0 and
    silently overwrite the previous batch's points."""
    points = [
        qm.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL,
                                         f"{video_id}:text:{start_idx + i}")),
                       vector=vec.tolist(), payload=payload)
        for i, (vec, payload) in enumerate(zip(vectors, payloads))
    ]
    if points:
        client().upsert(collection_name=TEXT_COLLECTION, points=points, wait=True)


def search_text(vector: np.ndarray, user_id: str, *, top_k: int,
                video_id: str | None = None,
                video_ids: list[str] | None = None) -> list[dict[str, Any]]:
    # Every kind. Document chunks (paper/deck/post) share this collection with
    # video transcripts and were excluded until Epic 4 could render a page, a
    # slide or an anchor instead of assuming a timestamp — a citation nobody
    # can follow being the same defect whatever the reason. All three lifted
    # together, with the visual branch, in one commit: lifting either alone
    # re-opens the failure the other one's guard was hiding.
    flt = _user_filter(user_id, video_id, video_ids)
    try:
        hits = client().query_points(
            collection_name=TEXT_COLLECTION,
            query=vector.tolist(),
            limit=top_k,
            query_filter=flt,
            with_payload=True,
            search_params=qm.SearchParams(
                quantization=qm.QuantizationSearchParams(rescore=True)
                if QDRANT_QUANTIZATION else None,
            ),
        ).points
    except Exception as exc:
        if "doesn't exist" in str(exc) or "Not found" in str(exc):
            return []
        raise
    return [{"score": float(h.score), **(h.payload or {})} for h in hits]


def delete_video(user_id: str, video_id: str) -> None:
    """Purge a video from BOTH branches (frames + transcript)."""
    sel = qm.FilterSelector(filter=_user_filter(user_id, video_id))
    for coll in (QDRANT_COLLECTION, TEXT_COLLECTION):
        try:
            client().delete(collection_name=coll, points_selector=sel, wait=True)
        except Exception:
            pass  # text collection may not exist if transcript is disabled


def collection_ready() -> bool:
    try:
        return client().collection_exists(QDRANT_COLLECTION)
    except Exception:
        return False
