"""Unit tests for the per-kind locator, label and deeplink (src/rag/search.py).

These run against synthetic hits, no stack and no model, because the thing
worth pinning is a set of *decisions* rather than a retrieval outcome:

  * a document never reports an `ms` it does not have — the failure this epic
    exists to stop is a citation that looks seekable and is not;
  * a deeplink is either somewhere a reader can actually land, or None. A
    fragment that silently does nothing and a URL on a hostname that resolves
    only inside the compose network are the same defect wearing a link.
"""
from __future__ import annotations

import pytest

from src.rag.search import (_label, _locator_deeplink, _locator_payload,
                            _public_url)

YT = {"source": "youtube", "url": "https://www.youtube.com/watch?v=abc123",
      "uri": "https://www.youtube.com/watch?v=abc123"}
POST = {"source": "post", "url": "https://blog.example.com/p/scaling",
        "uri": "http://substack-fixtures/scaling/scaling.md"}
FIXTURE_ONLY = {"source": "post", "url": None,
                "uri": "http://substack-fixtures/scaling/scaling.md"}
ARXIV = {"source": "paper", "url": None, "uri": "https://arxiv.org/pdf/1706.03762"}
UPLOADED = {"source": "paper", "url": None, "uri": "storage://papers/x/abc.pdf"}


# ── Which URL is publishable ─────────────────────────────────────────────────

def test_the_canonical_url_wins_over_the_fetch_uri() -> None:
    """A post is FETCHED from a fixture host and LIVES at its own URL. The
    citation must name the second."""
    assert _public_url(POST) == "https://blog.example.com/p/scaling"


def test_a_compose_network_host_is_not_a_public_url() -> None:
    """`http://substack-fixtures/...` resolves for the worker and for nobody
    else. Returning it would produce a link that 404s in the reader's browser
    — worse than admitting there is no link."""
    assert _public_url(FIXTURE_ONLY) is None


def test_a_public_fetch_uri_is_usable_when_there_is_no_canonical_url() -> None:
    """Papers are registered BY their public URL, so the uri is the address."""
    assert _public_url(ARXIV) == "https://arxiv.org/pdf/1706.03762"


def test_a_storage_key_is_not_a_url() -> None:
    assert _public_url(UPLOADED) is None


def test_no_metadata_row_yields_no_url() -> None:
    assert _public_url(None) is None
    assert _public_url({}) is None


# ── The locator, per kind ────────────────────────────────────────────────────

def test_a_video_locator_is_a_time_span() -> None:
    loc = _locator_payload("video", {"ms": 142500}, 142500)
    assert loc["start_ms"] == 142500
    assert loc["end_ms"] > loc["start_ms"], "a zero-length moment cannot be played"


def test_a_paper_locator_is_a_page() -> None:
    assert _locator_payload("paper", {"page": 4}, 0) == {"page": 4}


def test_a_deck_locator_is_a_slide() -> None:
    assert _locator_payload("deck", {"slide": 12}, 0) == {"slide": 12}


def test_a_post_locator_carries_the_anchor_and_whether_it_resolves() -> None:
    loc = _locator_payload("post", {"anchor": "what-breaks-first",
                                    "heading": "What breaks first",
                                    "anchor_native": True}, 0)
    assert loc == {"anchor": "what-breaks-first",
                   "heading": "What breaks first", "anchor_native": True}


def test_a_document_locator_never_invents_a_timestamp() -> None:
    """The whole point. `ms` is passed in for every kind because the caller
    computes it once — a document must ignore it rather than record it."""
    for kind, hit in (("paper", {"page": 2}), ("deck", {"slide": 3}),
                      ("post", {"anchor": "a", "anchor_native": True})):
        assert "start_ms" not in _locator_payload(kind, hit, 999_999)


# ── The human label ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind,loc,expected", [
    ("video", {"start_ms": 142_500}, "02:22"),
    ("paper", {"page": 4}, "p. 4"),
    ("deck", {"slide": 12}, "slide 12"),
    ("post", {"anchor": "why-now", "heading": "Why now"}, "§ Why now"),
])
def test_each_kind_reads_in_its_own_vocabulary(kind, loc, expected) -> None:
    assert _label(kind, loc) == expected


def test_the_opening_of_a_heading_less_post_reads_as_prose() -> None:
    """Four posts in the real corpus have `_top` as their ONLY anchor. `_top`
    is an implementation detail; a reader is told where they are."""
    assert _label("post", {"anchor": "_top", "heading": ""}) == "the opening"


def test_a_locator_with_a_missing_number_still_labels() -> None:
    """Degraded, not crashed: a payload written by an older ingest lacks the
    field, and a citation without a page is still a citation of that paper."""
    assert _label("paper", {}) == "paper"
    assert _label("deck", {}) == "deck"


# ── The deeplink ─────────────────────────────────────────────────────────────

def test_a_video_deeplink_seeks(monkeypatch) -> None:
    link = _locator_deeplink("video", YT, "yt_abc123", {"start_ms": 142_500})
    assert link.endswith("t=142")


def test_a_post_deeplink_is_the_url_plus_the_anchor() -> None:
    link = _locator_deeplink("post", POST, "doc_1",
                             {"anchor": "why-now", "anchor_native": True})
    assert link == "https://blog.example.com/p/scaling#why-now"


def test_a_synthesised_anchor_links_to_the_post_not_a_dead_fragment() -> None:
    """A bold pseudo-heading gives us a usable index key but no `id=` in the
    rendered page. Appending it produces a link that scrolls nowhere and looks
    broken; linking to the post is honest and still useful."""
    link = _locator_deeplink("post", POST, "doc_1",
                             {"anchor": "the-moat", "anchor_native": False})
    assert link == "https://blog.example.com/p/scaling"


def test_the_opening_of_a_post_links_to_the_post() -> None:
    link = _locator_deeplink("post", POST, "doc_1",
                             {"anchor": "_top", "anchor_native": True})
    # `_top` IS native (the top of a page always resolves) but there is no
    # `id="_top"` to jump to — the post's own URL already lands there, and
    # `#_top` would look like a working fragment that isn't.
    assert link == "https://blog.example.com/p/scaling"


def test_a_post_with_no_public_url_has_no_deeplink() -> None:
    """None, not a fixture URL. The UI renders an unclickable citation."""
    assert _locator_deeplink("post", FIXTURE_ONLY, "doc_1",
                             {"anchor": "why-now", "anchor_native": True}) is None


def test_a_paper_deeplink_opens_the_pdf_at_the_page() -> None:
    """#page=N is the PDF open-parameter browser viewers honour, and is
    harmless to a viewer that ignores it."""
    assert _locator_deeplink("paper", ARXIV, "doc_2", {"page": 4}) == \
        "https://arxiv.org/pdf/1706.03762#page=4"


def test_an_uploaded_paper_deeplinks_through_the_api() -> None:
    assert _locator_deeplink("paper", UPLOADED, "doc_2", {"page": 4}) == \
        "/api/document/doc_2#page=4"
