"""Cross-source blending (REC-316): one source or kind cannot own the answer.

Driven with synthetic windows rather than a live query, because the property
being pinned is a policy, not a retrieval outcome. The two failure modes it
guards are opposites, and a fix for one is the classic way to introduce the
other:

  * crowding — every citation from one post, because a 39-post corpus
    contributes thousands of text chunks against 31 videos' transcripts;
  * over-correction — a diversity rule that returns FEWER citations than it
    could have, or that reorders them so [1] is not the best match.
"""
from __future__ import annotations

from src.config import MAX_CITATIONS_PER_SOURCE
from src.rag.search import _blend


def w(kind: str, source: str, rrf: float) -> dict:
    return {"kind": kind, "video_id": source, "rrf": rrf}


def kinds(ws: list[dict]) -> list[str]:
    return [x["kind"] for x in ws]


def sources(ws: list[dict]) -> list[str]:
    return [x["video_id"] for x in ws]


# ── Crowding ─────────────────────────────────────────────────────────────────

def test_a_lower_scoring_source_is_never_crowded_out_entirely() -> None:
    """The measured case: a query about AI capex returned three windows from
    one post and two from another before any talk appeared.

    Note what is NOT asserted — that doc_a appears at most twice. With seven
    candidates for six slots, deferring doc_a's extra sections would mean
    returning four citations, and four citations where six were available is
    the worse answer. The property that matters is that both talks are in.
    """
    windows = [w("post", "doc_a", 0.9 - i / 100) for i in range(5)] + \
              [w("video", "yt_1", 0.4), w("video", "yt_2", 0.3)]
    out = _blend(windows, 6)
    assert "yt_1" in sources(out) and "yt_2" in sources(out)


def test_the_per_source_limit_binds_when_there_are_alternatives() -> None:
    """With enough distinct sources to fill every slot, the limit is a real
    limit rather than a preference — nothing has to be filled back in."""
    windows = [w("post", "doc_a", 0.99 - i / 100) for i in range(6)] + \
              [w("post", f"doc_{i}", 0.5) for i in range(6)]
    out = _blend(windows, 6)
    assert sources(out).count("doc_a") == MAX_CITATIONS_PER_SOURCE


def test_one_kind_cannot_crowd_out_another_that_is_relevant() -> None:
    windows = [w("post", f"doc_{i}", 0.9 - i / 100) for i in range(8)] + \
              [w("video", "yt_1", 0.2)]
    out = _blend(windows, 6)
    assert "video" in kinds(out), "a relevant talk was crowded out entirely"


def test_a_query_that_two_kinds_answer_cites_both() -> None:
    """The headline REC-316 promise, in its smallest form."""
    windows = [w("video", "yt_1", 0.9), w("video", "yt_1", 0.8),
               w("video", "yt_2", 0.7), w("paper", "doc_p", 0.6),
               w("post", "doc_b", 0.5)]
    out = _blend(windows, 3)
    assert len(set(kinds(out))) >= 2


def test_the_kind_limit_scales_with_top_k() -> None:
    """An absolute count sensible at k=6 never binds at k=3, which would make
    the promise above quietly false for any caller passing a smaller top_k."""
    windows = [w("post", f"doc_{i}", 0.9 - i / 100) for i in range(9)] + \
              [w("video", "yt_1", 0.1)]
    for k in (3, 6, 9):
        assert "video" in kinds(_blend(windows, k)), f"crowded out at k={k}"


# ── Over-correction ──────────────────────────────────────────────────────────

def test_a_single_source_answer_is_not_truncated() -> None:
    """When one video is the ONLY thing relevant, all six slots are still
    filled from it. The caps are a tie-break between candidates, not a quota
    that throws evidence away — an answer with two citations where six were
    available is a worse answer."""
    windows = [w("video", "yt_1", 0.9 - i / 100) for i in range(9)]
    out = _blend(windows, 6)
    assert len(out) == 6
    assert set(sources(out)) == {"yt_1"}


def test_fewer_candidates_than_slots_returns_them_all() -> None:
    windows = [w("video", "yt_1", 0.9), w("post", "doc_a", 0.8)]
    assert len(_blend(windows, 6)) == 2


def test_nothing_in_nothing_out() -> None:
    assert _blend([], 6) == []


# ── Ranking is never sacrificed for diversity ────────────────────────────────

def test_the_result_stays_ranked_best_first() -> None:
    """Diversity decides WHICH windows are cited, never in what order. A reader
    reading [1] before [2] is entitled to assume [1] matched better."""
    windows = [w("post", "doc_a", 0.99), w("post", "doc_a", 0.98),
               w("post", "doc_a", 0.97), w("video", "yt_1", 0.10)]
    out = _blend(windows, 4)
    scores = [x["rrf"] for x in out]
    assert scores == sorted(scores, reverse=True)


def test_the_single_best_window_is_always_cited() -> None:
    """No cap can be reached before the first pick, so the top-scoring window
    is unconditionally in the answer."""
    windows = [w("paper", "doc_p", 0.99)] + \
              [w("video", f"yt_{i}", 0.5) for i in range(10)]
    assert _blend(windows, 6)[0]["video_id"] == "doc_p"


def test_deferred_windows_come_back_in_score_order() -> None:
    """When the caps defer more than the free slots can take, the ones that
    return are the best of them — not whichever happened to be deferred first
    with a lower score."""
    windows = [w("post", "doc_a", 0.99), w("post", "doc_a", 0.98),
               w("post", "doc_a", 0.97), w("post", "doc_a", 0.50)]
    out = _blend(windows, 3)
    assert [x["rrf"] for x in out] == [0.99, 0.98, 0.97]
