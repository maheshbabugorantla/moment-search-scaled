"""Read path: question -> retrieve -> gate -> cited answer (or honest abstain).

Retrieval is milliseconds; the multimodal LLM call is seconds and dominates
cost. So the shape is a confidence funnel: fetch KNN_K candidates, collapse
temporal near-duplicates, trim to TOP_K, and — Gate 1 — if even the best
score is below CONFIDENCE_THRESHOLD, abstain WITHOUT calling the LLM. That
one free check kills most hallucination risk. Generated answers get their
[n] citations validated; invented references are stripped.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .. import config, db, llm, storage
from ..config import (BRANCH_TOP_K, CONFIDENCE_THRESHOLD, CROSS_MODAL_BOOST,
                      FUSION_WINDOW_S, RRF_K, TEXT_CONFIDENCE_THRESHOLD, TOP_ANCHOR,
                      TOP_K)
from . import vector_store
from .embeddings import embed_query, embed_text

ABSTAIN = ("I couldn't find that in your sources — nothing indexed looks "
           "related to the question, in any talk, paper or post.")


def _seconds(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def _fuse(visual_hits: list[dict], text_hits: list[dict]) -> list[dict]:
    """Reciprocal-Rank-Fusion of the two branches into locator windows.

    Raw scores are incomparable (CLIP ~0.3 vs bge ~0.7), so we rank each branch
    on its own and score by rank: rrf = 1/(RRF_K + rank). Then hits are
    bucketed into windows, the rrf of the best hit per modality is summed, and
    windows where BOTH modalities agree are boosted — two independent signals
    pointing at the same place is the strongest evidence.

    Window identity is (kind, source, locator), because "the same moment"
    means something different per kind (REC-332):
      * video — hits within FUSION_WINDOW_S seconds of each other in the same
        video. Time-proximity matching, first-hit-anchored — bit-for-bit the
        pre-multi-source behaviour, since video hits carry no `kind`.
      * paper — hits on the same page of the same paper. Pages have no time
        (a paper hit's `t` falls back to 0.0), so without this key every page
        of one paper would land in one t=0 window and only the best chunk
        would survive — one citation per paper, no matter how many distinct
        pages matched.
      * deck — hits on the same slide of the same deck. A slide with both
        extracted text and a vision caption gets the same cross-modal boost a
        frame+transcript match gets: that is a decision, not an accident —
        two independent readings of one slide agreeing IS the same signal.
      * post — hits under the same heading anchor of the same post. Section
        level, not chunk level, deliberately: a long section splits into
        several chunks that all cite ONE anchor, so collapsing them is
        correct, not the bug this key exists to fix. Two chunks of one section
        are one citation; two sections are two.
    """
    def ranked(hits, modality):
        out = []
        for rank, h in enumerate(hits):
            t = float(h.get("t_start", h.get("ms", 0) / 1000.0))
            out.append({**h, "modality": modality, "rrf": 1.0 / (RRF_K + rank), "t": t})
        return out

    def _locator(h) -> tuple:
        """The non-time window key for document kinds; None for video."""
        kind = h.get("kind") or "video"
        if kind == "paper":
            return (kind, h["video_id"], h.get("page"))
        if kind == "deck":
            return (kind, h["video_id"], h.get("slide"))
        if kind == "post":
            return (kind, h["video_id"], h.get("anchor"))
        return ()  # video: matched by time proximity below, not by this key

    windows: list[dict] = []
    # Hits arrive best-first (rrf desc), so the first hit landing in a window for
    # a given modality is that modality's best hit there.
    for h in sorted(ranked(visual_hits, "frame") + ranked(text_hits, "text"),
                    key=lambda x: x["rrf"], reverse=True):
        loc = _locator(h)
        if loc:
            w = next((w for w in windows if w["locator"] == loc), None)
        else:
            w = next((w for w in windows if not w["locator"]
                      and w["video_id"] == h["video_id"]
                      and abs(w["t"] - h["t"]) <= FUSION_WINDOW_S), None)
        if w is None:
            w = {"kind": h.get("kind") or "video", "video_id": h["video_id"],
                 "t": h["t"], "locator": loc, "rrf": 0.0,
                 "modalities": set(), "frame": None, "text": None}
            windows.append(w)
        w["modalities"].add(h["modality"])
        slot = "frame" if h["modality"] == "frame" else "text"
        # Keep only the BEST hit per modality. Summing every hit would let a
        # burst of near-identical frames clustered in one 15s window inflate its
        # score past a genuine frame+transcript match — the bug that ranked a
        # silent frame-burst above the moment that actually answered.
        if w[slot] is None:
            w[slot] = h
    for w in windows:
        # Score = best frame + best transcript hit; ×boost when BOTH modalities
        # agree at this instant (two independent signals = strongest evidence).
        w["rrf"] = (w["frame"]["rrf"] if w["frame"] else 0.0) + \
                   (w["text"]["rrf"] if w["text"] else 0.0)
        if {"frame", "text"} <= w["modalities"]:
            w["rrf"] *= CROSS_MODAL_BOOST
    windows.sort(key=lambda w: w["rrf"], reverse=True)
    return windows


def _deeplink(video: dict | None, video_id: str, ms: int) -> str:
    secs = ms // 1000
    if video and video.get("source") == "youtube" and video.get("url"):
        sep = "&" if "?" in video["url"] else "?"
        return f"{video['url']}{sep}t={secs}"
    return f"/api/video/{video_id}#t={secs}"


# ── Locators: one scheme per kind ────────────────────────────────────────────
#
# A citation is only worth anything if its locator points at something a reader
# can reach. Each kind names its position differently and each deeplinks
# differently, so the three concerns — the structured locator, the human label,
# and the URL — are derived together, per kind, from the winning window.
#
# The window carries the locator tuple _fuse() keyed on, so nothing here has to
# re-derive which page/slide/anchor won: it reads the hits it already chose.

def _public_url(meta: dict | None) -> str | None:
    """Where a READER can find this source, or None.

    `url` is the canonical address (a YouTube watch URL, a post's front-matter
    URL). `uri` is where INGESTION fetched the bytes and is only publishable
    when it happens to be a public http(s) address — an arXiv PDF link is;
    `storage://papers/...` and `http://substack-fixtures/...` are not.
    """
    if not meta:
        return None
    url = (meta.get("url") or "").strip()
    if url.startswith(("http://", "https://")):
        return url
    uri = (meta.get("uri") or "").strip()
    # A compose-network hostname has no dot and resolves for nobody outside the
    # network; treating it as public is how a citation gets a dead deeplink.
    if uri.startswith("https://") or (uri.startswith("http://")
                                      and "." in uri.split("/")[2].split(":")[0]):
        return uri
    return None


def _locator_payload(kind: str, hit: dict, ms: int) -> dict:
    """The structured locator, exactly as REC-314 specifies it per kind."""
    if kind == "paper":
        return {"page": hit.get("page")}
    if kind == "deck":
        return {"slide": hit.get("slide")}
    if kind == "post":
        return {"anchor": hit.get("anchor"),
                "heading": hit.get("heading"),
                # Whether {url}#{anchor} actually scrolls the live page. A
                # synthesised anchor (a bold pseudo-heading) and the `_top`
                # of a heading-less post are real locators for OUR index and
                # honest citations, but no renderer will jump to them — so
                # the UI links to the post itself rather than a fragment that
                # silently does nothing.
                "anchor_native": bool(hit.get("anchor_native"))}
    return {"start_ms": ms, "end_ms": ms + int(FUSION_WINDOW_S * 1000)}


def _label(kind: str, loc: dict) -> str:
    """What a human reads in place of a timestamp."""
    if kind == "paper":
        return f"p. {loc['page']}" if loc.get("page") is not None else "paper"
    if kind == "deck":
        return f"slide {loc['slide']}" if loc.get("slide") is not None else "deck"
    if kind == "post":
        heading = (loc.get("heading") or "").strip()
        if heading:
            return f"§ {heading}"
        # `_top` is the opening of a post that starts with prose. "the opening"
        # is what it means; "_top" is an implementation detail.
        return "the opening" if loc.get("anchor") == TOP_ANCHOR else "post"
    return _seconds(loc["start_ms"])


def _locator_deeplink(kind: str, meta: dict | None, source_id: str,
                      loc: dict) -> str | None:
    """Where clicking the citation lands. None when nothing can be reached —
    an honest absence the UI renders as an unclickable citation, rather than a
    link that 404s."""
    url = _public_url(meta)
    if kind == "video":
        return _deeplink(meta, source_id, loc["start_ms"])
    if kind == "post":
        if not url:
            return None
        anchor = loc.get("anchor")
        # `_top` is native in the sense that the top of a page always resolves,
        # but there is no `id="_top"` to jump to — the bare URL already lands
        # there, and `#_top` would look like a working fragment that isn't.
        if anchor and anchor != TOP_ANCHOR and loc.get("anchor_native"):
            return f"{url}#{anchor}"
        return url
    if kind in ("paper", "deck"):
        page = loc.get("page") if kind == "paper" else loc.get("slide")
        # #page=N is the PDF open-parameter every browser viewer honours; it is
        # also harmless on a viewer that ignores it, which is why it can be
        # appended to a bare arXiv link without checking.
        base = url or f"/api/document/{source_id}"
        return f"{base}#page={page}" if page is not None else base
    return None


def _thumb_url(user_id: str, video_id: str, idx: int) -> str:
    """Browser-facing thumbnail URL. Presigned GET straight to the bucket when
    the provider supports it (an <img> tag can't send auth headers); the API
    serves the bytes itself only in local-dev mode."""
    if storage.presign_capable():
        return storage.presign_get(storage.frame_key(user_id, video_id, idx))
    return f"/api/frame/{video_id}/{idx:06d}.jpg?u={user_id}"


def _media_url(video: dict | None, user_id: str, video_id: str) -> str | None:
    """Playback URL for uploaded videos (YouTube plays via its own URL)."""
    if not video or video.get("source") != "upload" or not video.get("storage_key"):
        return None
    if storage.presign_capable():
        return storage.presign_get(video["storage_key"])
    return f"/api/video/{video_id}?u={user_id}"


def retrieve(question: str, user_id: str, *, top_k: int | None = None,
             video_id: str | None = None,
             video_ids: list[str] | None = None) -> dict[str, Any]:
    """Multimodal retrieve: query BOTH branches (CLIP frames + transcript text),
    fuse by RRF into time windows, and return numbered moment-citations.

    Returns {citations, best_visual, best_text} — the two raw bests feed the
    confidence gate (RRF scores are too small to threshold on). video_ids scopes
    the search to chosen videos (UI select/unselect)."""
    k = top_k or TOP_K

    # Visual branch — CLIP text→image.
    vhits = vector_store.search(embed_text(question), user_id, top_k=BRANCH_TOP_K,
                                video_id=video_id, video_ids=video_ids)
    best_visual = vhits[0]["score"] if vhits else 0.0

    # Text branch — bge query→transcript-chunk (only if transcript is enabled).
    thits: list[dict] = []
    best_text = 0.0
    if config.ENABLE_TRANSCRIPT:
        thits = vector_store.search_text(embed_query(question), user_id,
                                         top_k=BRANCH_TOP_K, video_id=video_id,
                                         video_ids=video_ids)
        best_text = thits[0]["score"] if thits else 0.0

    windows = _fuse(vhits, thits)[:k]
    videos = db.videos_by_ids(sorted({w["video_id"] for w in windows}))
    citations = []
    for i, w in enumerate(windows, 1):
        vid = w["video_id"]
        meta = videos.get(vid)
        fr, tx = w["frame"], w["text"]
        kind = w["kind"]
        # Which hit describes the location differs by kind, and getting it
        # backwards degrades quietly rather than crashing. A video window
        # prefers its FRAME — that timestamp is the precise visual seek. A
        # document window prefers its TEXT: a post image payload carries an
        # `anchor` but no heading and no anchor_native (it is a frame_key
        # point, not a chunk), so building the locator from it produced a
        # citation labelled "post" that linked to the article root instead of
        # "§ AI's $600B Question" linking to the section.
        src = (fr or tx) if kind == "video" else (tx or fr)
        loc = _locator_payload(kind, src or {},
                               # video only: the frame's exact timestamp when
                               # there is one (precise visual seek), else the
                               # transcript chunk's start.
                               int(fr["ms"]) if fr and fr.get("ms") is not None
                               else int(w["t"] * 1000))
        idx = int(fr["idx"]) if fr and fr.get("idx") is not None else None
        label = _label(kind, loc)
        citations.append({
            "n": i,
            # `sourceId`/`kind`/`locator` are the REC-314 payload; `video_id`
            # and the flat fields below stay for the video UI that predates it.
            "sourceId": vid,
            "kind": kind,
            "locator": loc,
            "label": label,
            "video_id": vid,
            "title": (meta or {}).get("title") or vid,
            "url": _public_url(meta),
            "source": (meta or {}).get("source"),
            # A document has no timestamp. Null rather than a plausible-looking
            # 00:00, which is a lie a reader would act on.
            "ms": loc["start_ms"] if kind == "video" else None,
            "timestamp": label if kind == "video" else None,
            "idx": idx,
            "thumbnail": _thumb_url(user_id, vid, idx) if idx is not None else None,
            "media_url": _media_url(meta, user_id, vid),
            "deeplink": _locator_deeplink(kind, meta, vid, loc),
            "score": round(w["rrf"], 4),
            "transcript": (tx or {}).get("text"),
            "modalities": sorted(w["modalities"]),
        })
    return {"citations": citations, "best_visual": best_visual, "best_text": best_text}


def _fallback_answer(citations: list[dict[str, Any]]) -> str:
    """No-LLM summary: rank the visually-closest moments. Honest about being
    similarity, not synthesis."""
    top = citations[0]
    where = f"{top['title']} at {top['label']}" if top.get("title") else top["label"]
    others = ", ".join(f"{c['label']} [{c['n']}]" for c in citations[1:4])
    msg = f"Closest match: {where} [{top['n']}] (similarity {top['score']})."
    if others:
        msg += f" Other relevant moments: {others}."
    return msg


_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _validate_citations(answer: str, n_frames: int) -> str:
    """Strip invented [n] references the model has no frame for."""
    def fix(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        valid = [str(x) for x in nums if 1 <= x <= n_frames]
        return f"[{', '.join(valid)}]" if valid else ""
    return _CITE_RE.sub(fix, answer)


def _build_moments(user_id: str, citations: list[dict[str, Any]]) -> list[dict]:
    """Turn citations into what the LLM sees: each moment carries its frame
    image (if any) and/or its transcript excerpt (if any), numbered to match."""
    def frame_bytes(c):
        if c.get("idx") is None:
            return None
        try:
            return storage.get_bytes(storage.frame_key(user_id, c["video_id"], c["idx"]))
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=6) as ex:
        images = list(ex.map(frame_bytes, citations))
    # `where` rather than `timestamp`: the model must be told a paper chunk is
    # at "p. 4", not at "00:00". Handing every document a fake timestamp is how
    # a model learns to write one back out.
    return [{"image": img, "transcript": c.get("transcript"),
             "kind": c["kind"], "title": c.get("title"),
             "where": c["label"], "timestamp": c["label"]}
            for img, c in zip(images, citations)]


def resolve_llm(user_id: str) -> tuple[llm.LLMConfig | None, str]:
    """Which model answers for this tenant: their own hosted endpoint
    (ms_user_llms — e.g. a vLLM server) first, the server-wide LLM_* env
    config as fallback. Returns (config, source) with source in
    {"user", "server", "none"}."""
    row = db.get_user_llm(user_id)
    if row and row.get("model"):
        return llm.from_row(row), "user"
    cfg = llm.env_config()
    return (cfg, "server") if cfg else (None, "none")


def ask(question: str, user_id: str, *, top_k: int | None = None,
        video_id: str | None = None,
        video_ids: list[str] | None = None) -> dict[str, Any]:
    r = retrieve(question, user_id, top_k=top_k, video_id=video_id, video_ids=video_ids)
    citations = r["citations"]
    result: dict[str, Any] = {"question": question, "citations": citations}

    if not citations:
        result.update(answer="Nothing relevant was found. Try adding a source first.",
                      llm_used=False, abstained=True)
        return result

    # Gate 1 — confidence on the RAW per-branch bests (not the RRF score).
    # Abstain only if NEITHER what's on screen nor what's said looks relevant.
    visual_ok = r["best_visual"] >= CONFIDENCE_THRESHOLD
    text_ok = r["best_text"] >= TEXT_CONFIDENCE_THRESHOLD
    if CONFIDENCE_THRESHOLD and not visual_ok and not text_ok:
        result.update(answer=ABSTAIN, llm_used=False, abstained=True)
        return result

    cfg, source = resolve_llm(user_id)
    if cfg is None:
        # No generative model — summarize the best matches instead of inventing.
        result.update(answer=_fallback_answer(citations), llm_used=False,
                      note=("Retrieval-only results. Connect your own model "
                            "(vLLM/Ollama/API) in settings, or set LLM_API_KEY "
                            "on the server, for a synthesized, grounded answer."))
        return result

    moments = _build_moments(user_id, citations)
    result["answer"] = _validate_citations(llm.answer(question, moments, cfg),
                                           len(citations))
    result["llm_used"] = True
    result["llm_source"] = source          # "user" = their own hosted model
    result["llm_model"] = cfg.model
    return result
