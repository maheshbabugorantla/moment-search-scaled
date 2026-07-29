"""Unit tests for _fuse() window identity (REC-332).

The fix under test: window identity is (kind, source, locator) — video keeps
time-proximity bucketing bit-for-bit (its hits carry no `kind`), a paper
buckets per page, a deck per slide. Before this, every page of a paper fell
into one t=0 window and only one chunk survived: one citation per paper,
however many distinct pages matched.

These drive _fuse() directly with synthetic hits (imports src — the tests
container sets PYTHONPATH=/app, precedented by test_error_envelope.py).
End-to-end cross-source assertions belong to Epic 4; this pins the fusion
identity alone.
"""
from __future__ import annotations

from src.rag.search import _fuse


def _vhit(video_id: str, t: float, score: float = 0.3) -> dict:
    """A CLIP frame hit as vector_store.search returns it."""
    return {"score": score, "user_id": "u", "video_id": video_id,
            "ms": int(t * 1000), "idx": int(t), "modality": "frame",
            "t_start": t, "t_end": t}


def _thit(video_id: str, t: float, score: float = 0.6) -> dict:
    """A transcript chunk hit — video kind, so NO `kind` field (pre-Epic-2
    points never carried one; absence must keep meaning video)."""
    return {"score": score, "user_id": "u", "video_id": video_id,
            "ms": int(t * 1000), "t_start": t, "t_end": t + 20.0,
            "text": f"spoken at {t}"}


def _phit(doc_id: str, page: int, score: float = 0.6) -> dict:
    """A paper chunk hit — kind: paper, page locator, and NO timestamp
    (t falls back to 0.0, the exact collision the re-key fixes)."""
    return {"score": score, "user_id": "u", "video_id": doc_id,
            "kind": "paper", "page": page, "modality": "text",
            "text": f"page {page} text"}


def _dhit(doc_id: str, slide: int, score: float = 0.6) -> dict:
    return {"score": score, "user_id": "u", "video_id": doc_id,
            "kind": "deck", "slide": slide, "modality": "text",
            "text": f"slide {slide} text"}


# ── Video behaviour is unchanged ─────────────────────────────────────────────

def test_video_hits_within_the_window_still_merge() -> None:
    windows = _fuse([_vhit("yt_a", 10.0)], [_thit("yt_a", 18.0)])
    assert len(windows) == 1
    assert windows[0]["modalities"] == {"frame", "text"}


def test_video_hits_beyond_the_window_still_split() -> None:
    windows = _fuse([_vhit("yt_a", 10.0)], [_thit("yt_a", 300.0)])
    assert len(windows) == 2


def test_video_ranking_is_unchanged_by_the_rekey() -> None:
    """The golden case: two videos, one cross-modal moment, one frame-only
    moment. The boosted cross-modal window must rank first — exactly the
    pre-change ordering."""
    windows = _fuse(
        [_vhit("yt_a", 10.0), _vhit("yt_b", 50.0)],
        [_thit("yt_a", 12.0)],
    )
    assert [(w["video_id"], round(w["t"])) for w in windows] == \
        [("yt_a", 10), ("yt_b", 50)]
    assert windows[0]["modalities"] == {"frame", "text"}
    assert windows[1]["modalities"] == {"frame"}


def test_video_windows_carry_the_new_fields_with_defaults() -> None:
    (w,) = _fuse([_vhit("yt_a", 10.0)], [])
    assert w["kind"] == "video"
    assert not w["locator"]


# ── The paper fix ────────────────────────────────────────────────────────────

def test_three_matching_pages_yield_three_windows() -> None:
    """The headline: one paper, three distinct matching pages, three windows —
    not one t=0 window that discards two of them."""
    windows = _fuse([], [_phit("doc_x", 2), _phit("doc_x", 7), _phit("doc_x", 11)])
    assert len(windows) == 3
    assert {w["locator"][2] for w in windows} == {2, 7, 11}
    assert all(w["kind"] == "paper" for w in windows)


def test_two_hits_on_the_same_page_share_a_window() -> None:
    """Within one page, only the best chunk survives — same dedup rule as a
    frame burst inside one video window."""
    windows = _fuse([], [_phit("doc_x", 2, 0.9), _phit("doc_x", 2, 0.4)])
    assert len(windows) == 1
    assert windows[0]["text"]["score"] == 0.9


def test_two_papers_do_not_collide_on_the_same_page_number() -> None:
    windows = _fuse([], [_phit("doc_x", 3), _phit("doc_y", 3)])
    assert len(windows) == 2
    assert {w["video_id"] for w in windows} == {"doc_x", "doc_y"}


def test_a_paper_at_t_zero_does_not_swallow_a_video_at_t_zero() -> None:
    """Both land at t=0.0; the kind in the key keeps them apart."""
    windows = _fuse([_vhit("yt_a", 0.0)], [_phit("doc_x", 1)])
    assert len(windows) == 2
    assert {w["kind"] for w in windows} == {"video", "paper"}


def test_paper_and_video_hits_coexist_ranked_by_rrf() -> None:
    windows = _fuse(
        [_vhit("yt_a", 10.0)],
        [_thit("yt_a", 12.0), _phit("doc_x", 4), _phit("doc_x", 9)],
    )
    kinds = {w["kind"] for w in windows}
    assert kinds == {"video", "paper"}
    assert len([w for w in windows if w["kind"] == "paper"]) == 2
    # The cross-modal video window outranks single-modality paper pages.
    assert windows[0]["kind"] == "video"


# ── Decks bucket per slide, and the boost decision holds ─────────────────────

def test_deck_slides_bucket_separately() -> None:
    windows = _fuse([], [_dhit("deck_z", 1), _dhit("deck_z", 5)])
    assert len(windows) == 2
    assert {w["locator"][2] for w in windows} == {1, 5}


def test_a_slide_with_text_and_caption_gets_the_cross_modal_boost() -> None:
    """The written-down decision from REC-332: a deck slide whose extracted
    text AND vision caption both match gets the same boost a frame+transcript
    agreement gets — two independent readings of one slide agreeing is the
    same signal. A caption arrives through the visual branch as a `frame`
    modality hit carrying the slide locator."""
    caption_hit = {"score": 0.4, "user_id": "u", "video_id": "deck_z",
                   "kind": "deck", "slide": 3, "text": "caption of slide 3"}
    windows = _fuse([caption_hit], [_dhit("deck_z", 3)])
    assert len(windows) == 1
    assert windows[0]["modalities"] == {"frame", "text"}
    # Boosted above what either single-modality rrf could be alone.
    solo = _fuse([], [_dhit("deck_z", 3)])[0]["rrf"]
    assert windows[0]["rrf"] > 2 * solo * 0.9
