"""Recall@k gate for PR #8 (paper ingestion) and PR #9 (fusion re-key).

Runs INSIDE the api container (needs src + qdrant creds). Two things it proves:

  1. Retrieval recall  — for each labeled query, does the correct paper's
     relevant page appear in the top-k text hits? This is the PR #8 gate: if
     ingestion produced garbage chunks, recall collapses.
  2. Fusion behaviour   — the same hits are fused twice, by the OLD time-window
     _fuse and the NEW (kind, source, locator) _fuse (both implemented here so
     one run compares them without swapping images), reporting distinct-citation
     counts. This is the PR #9 gate: old fusion must collapse a paper to ONE
     citation; new fusion must surface the matching pages separately, while
     video ordering is unaffected (covered separately by the baseline diff).

Ground truth is (doc_id, set_of_acceptable_pages) where the page set is
DERIVED from the indexed chunks by exact-phrase match — never hand-asserted.
A query whose anchor phrase isn't found is reported as UNGRADEABLE and excluded
from the score rather than silently counted as a miss.

    python /app/eval/eval_recall.py [--k 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

sys.path.insert(0, "/app")

from qdrant_client.http import models as qm  # noqa: E402

from src import config, db  # noqa: E402
from src.rag import vector_store  # noqa: E402
from src.rag.embeddings import embed_query, embed_text  # noqa: E402

USER = config.DEFAULT_USER_ID
RRF_K, WINDOW_S, BOOST = config.RRF_K, config.FUSION_WINDOW_S, config.CROSS_MODAL_BOOST

# (query, paper title substring, anchor phrase that must appear in a relevant
# chunk). The anchor phrase pins ground truth to text that is provably indexed.
QUERIES = [
    ("What is scaled dot-product attention?", "Attention Is All You Need", "scaled dot-product attention"),
    ("How many attention heads does the transformer use?", "Attention Is All You Need", "parallel attention layers"),
    ("What is the formula for positional encoding?", "Attention Is All You Need", "positional encoding"),
    ("How does model performance scale with parameters and compute?", "Scaling Laws", "power-law"),
    ("Does model shape matter less than scale?", "Scaling Laws", "width"),
    ("What is the ARC dataset for measuring intelligence?", "On the Measure of Intelligence", "ARC"),
    ("How should intelligence be defined as skill acquisition efficiency?", "On the Measure of Intelligence", "skill-acquisition efficiency"),
    ("What is the era of experience in reinforcement learning?", "Era of Experience", "experience"),
    ("How does AlphaFold predict protein structure from sequence?", "AlphaFold", "structure"),
    ("What is the Evoformer architecture?", "AlphaFold", "Evoformer"),
    ("How does few-shot learning work with large language models?", "GPT-3", "few-shot"),
    ("What is masked language model pretraining?", "BERT", "masked"),
    ("How is reinforcement learning from human feedback used to align models?", "InstructGPT", "human feedback"),
    ("How does contrastive pretraining connect images and text?", "CLIP", "contrastive"),
]


def papers() -> dict[str, dict]:
    return {r["id"]: r for r in db.list_sources(USER, kind="paper", limit=100)
            if r["status"] == "indexed"}


def chunks_of(doc_id: str) -> list[dict]:
    """Every indexed chunk of a paper, from Qdrant (the source of truth for
    what retrieval can actually see)."""
    out, offset = [], None
    flt = qm.Filter(must=[qm.FieldCondition(key="video_id",
                                            match=qm.MatchValue(value=doc_id))])
    while True:
        pts, offset = vector_store.client().scroll(
            config.TEXT_COLLECTION, scroll_filter=flt, limit=256,
            offset=offset, with_payload=True)
        out += [p.payload for p in pts]
        if offset is None:
            return out


def ground_truth(rows: dict, chunk_cache: dict) -> list[dict]:
    """Resolve each labeled query to (doc_id, acceptable pages) by finding the
    anchor phrase in the actually-indexed chunks."""
    cases = []
    for question, title_sub, phrase in QUERIES:
        match = [r for r in rows.values() if title_sub.lower() in (r["title"] or "").lower()]
        if not match:
            cases.append({"q": question, "status": "UNGRADEABLE",
                          "why": f"no indexed paper matching {title_sub!r}"})
            continue
        doc = match[0]["id"]
        pages = sorted({c["page"] for c in chunk_cache[doc]
                        if phrase.lower() in (c.get("text") or "").lower()})
        if not pages:
            cases.append({"q": question, "status": "UNGRADEABLE",
                          "why": f"anchor phrase {phrase!r} not found in {doc} chunks"})
            continue
        cases.append({"q": question, "doc": doc, "pages": set(pages),
                      "title": match[0]["title"], "status": "OK"})
    return cases


def search_text_unfiltered(vec, k: int) -> list[dict]:
    """Text-branch search WITHOUT the kind guard — papers are excluded from
    /api/ask until Epic 4, so the eval queries the branch directly."""
    hits = vector_store.client().query_points(
        collection_name=config.TEXT_COLLECTION, query=vec.tolist(), limit=k,
        query_filter=qm.Filter(must=[qm.FieldCondition(
            key="user_id", match=qm.MatchValue(value=USER))]),
        with_payload=True).points
    return [{"score": float(h.score), **(h.payload or {})} for h in hits]


def _ranked(hits, modality):
    out = []
    for rank, h in enumerate(hits):
        t = float(h.get("t_start", h.get("ms", 0) / 1000.0))
        out.append({**h, "modality": modality, "rrf": 1.0 / (RRF_K + rank), "t": t})
    return out


def fuse_old(vhits, thits):
    """main's fusion: time windows only."""
    windows = []
    for h in sorted(_ranked(vhits, "frame") + _ranked(thits, "text"),
                    key=lambda x: x["rrf"], reverse=True):
        w = next((w for w in windows if w["video_id"] == h["video_id"]
                  and abs(w["t"] - h["t"]) <= WINDOW_S), None)
        if w is None:
            w = {"video_id": h["video_id"], "t": h["t"], "rrf": 0.0,
                 "modalities": set(), "frame": None, "text": None}
            windows.append(w)
        w["modalities"].add(h["modality"])
        slot = "frame" if h["modality"] == "frame" else "text"
        if w[slot] is None:
            w[slot] = h
    for w in windows:
        w["rrf"] = (w["frame"]["rrf"] if w["frame"] else 0.0) + \
                   (w["text"]["rrf"] if w["text"] else 0.0)
        if {"frame", "text"} <= w["modalities"]:
            w["rrf"] *= BOOST
    windows.sort(key=lambda w: w["rrf"], reverse=True)
    return windows


def fuse_new(vhits, thits):
    """PR #9's fusion: (kind, source, locator)."""
    def loc(h):
        kind = h.get("kind") or "video"
        if kind == "paper":
            return (kind, h["video_id"], h.get("page"))
        if kind == "deck":
            return (kind, h["video_id"], h.get("slide"))
        return ()

    windows = []
    for h in sorted(_ranked(vhits, "frame") + _ranked(thits, "text"),
                    key=lambda x: x["rrf"], reverse=True):
        l = loc(h)
        if l:
            w = next((w for w in windows if w["locator"] == l), None)
        else:
            w = next((w for w in windows if not w["locator"]
                      and w["video_id"] == h["video_id"]
                      and abs(w["t"] - h["t"]) <= WINDOW_S), None)
        if w is None:
            w = {"kind": h.get("kind") or "video", "video_id": h["video_id"],
                 "t": h["t"], "locator": l, "rrf": 0.0, "modalities": set(),
                 "frame": None, "text": None}
            windows.append(w)
        w["modalities"].add(h["modality"])
        slot = "frame" if h["modality"] == "frame" else "text"
        if w[slot] is None:
            w[slot] = h
    for w in windows:
        w["rrf"] = (w["frame"]["rrf"] if w["frame"] else 0.0) + \
                   (w["text"]["rrf"] if w["text"] else 0.0)
        if {"frame", "text"} <= w["modalities"]:
            w["rrf"] *= BOOST
    windows.sort(key=lambda w: w["rrf"], reverse=True)
    return windows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    k = args.k

    rows = papers()
    print(f"indexed papers: {len(rows)}")
    cache = {d: chunks_of(d) for d in rows}
    for d, cs in cache.items():
        print(f"  {rows[d]['title'][:45]:47s} {len(cs):4d} chunks, "
              f"pages 1-{max((c.get('page') or 0) for c in cs) if cs else 0}")

    cases = ground_truth(rows, cache)
    gradeable = [c for c in cases if c["status"] == "OK"]
    for c in cases:
        if c["status"] != "OK":
            print(f"  UNGRADEABLE: {c['q'][:50]!r} — {c['why']}")

    hits_doc = hits_page = 0
    collapse_old = collapse_new = 0
    rows_out = []
    for c in gradeable:
        vec = embed_query(c["q"])
        thits = search_text_unfiltered(vec, k)
        top_docs = [h["video_id"] for h in thits]
        doc_hit = c["doc"] in top_docs
        page_hit = any(h["video_id"] == c["doc"] and h.get("page") in c["pages"]
                       for h in thits)
        hits_doc += doc_hit
        hits_page += page_hit

        # Fusion comparison on the SAME hits (visual branch queried too, so the
        # video side of fusion is exercised exactly as production does it).
        vhits = vector_store.search(embed_text(c["q"]), USER, top_k=k)
        wo = fuse_old(vhits, thits)
        wn = fuse_new(vhits, thits)
        cited_old = len([w for w in wo if w["video_id"] == c["doc"]])
        cited_new = len([w for w in wn if w["video_id"] == c["doc"]])
        collapse_old += cited_old
        collapse_new += cited_new

        rank = next((i + 1 for i, h in enumerate(thits)
                     if h["video_id"] == c["doc"] and h.get("page") in c["pages"]), None)
        rows_out.append({"q": c["q"], "doc_hit": doc_hit, "page_hit": page_hit,
                         "rank": rank, "old_windows": cited_old,
                         "new_windows": cited_new})
        print(f"{'PASS' if page_hit else ('DOC ' if doc_hit else 'MISS')} "
              f"rank={str(rank):>4}  old={cited_old} new={cited_new}  {c['q'][:58]}")

    n = len(gradeable)
    print("\n" + "=" * 72)
    print(f"gradeable queries:        {n} of {len(QUERIES)}")
    print(f"recall@{k} (right paper):  {hits_doc}/{n} = {hits_doc / n:.1%}")
    print(f"recall@{k} (right page):   {hits_page}/{n} = {hits_page / n:.1%}")
    print(f"paper citations, OLD fusion: {collapse_old} total "
          f"({collapse_old / n:.2f}/query)   <- collapses to ~1")
    print(f"paper citations, NEW fusion: {collapse_new} total "
          f"({collapse_new / n:.2f}/query)   <- pages surface separately")
    print("=" * 72)
    json.dump(rows_out, open("/tmp/eval_results.json", "w"), indent=1)

    # Gate: page-level recall@10 must clear 80%, and new fusion must strictly
    # beat old on distinct paper citations.
    ok = (hits_page / n >= 0.8) and (collapse_new > collapse_old)
    print("GATE: PASS" if ok else "GATE: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
