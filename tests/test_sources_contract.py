"""GET /admin/sources — the unified status view.

How an admin knows a backfill is progressing, and what the resilience harness
polls to decide a run reached a terminal state. Read-only: it renders whatever
the manifest holds.
"""
from __future__ import annotations

import httpx
import pytest

from conftest import CORPUS_IDS

PAPER_URI = "https://example.com/sources-test-paper.pdf"


@pytest.fixture
def a_pending_paper(client: httpx.Client, auth: dict):
    """A document in the manifest, removed afterwards."""
    r = client.post("/admin/documents",
                    json={"uri": PAPER_URI, "kind": "paper", "title": "RAG Survey"},
                    headers=auth)
    assert r.status_code == 202
    doc_id = r.json()["id"]
    yield doc_id
    client.delete(f"/api/videos/{doc_id}", headers=auth)


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_sources_without_token_is_401(client: httpx.Client) -> None:
    assert client.get("/admin/sources").status_code == 401


def test_sources_with_wrong_token_is_401(client: httpx.Client) -> None:
    r = client.get("/admin/sources", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


# ── The unified view ──────────────────────────────────────────────────────────

_SOURCE_FIELDS = {"id", "kind", "status", "title", "pct", "uri", "error",
                  "created_at", "updated_at"}


def test_every_source_carries_the_documented_fields(
    client: httpx.Client, auth: dict
) -> None:
    body = client.get("/admin/sources", headers=auth).json()
    assert "sources" in body
    assert body["sources"]
    for s in body["sources"]:
        assert set(s) == _SOURCE_FIELDS
        assert s["kind"] in ("video", "paper", "deck")
        assert isinstance(s["pct"], int)


def test_videos_appear_as_kind_video(client: httpx.Client, auth: dict) -> None:
    body = client.get("/admin/sources", headers=auth).json()
    by_id = {s["id"]: s for s in body["sources"]}
    for vid in CORPUS_IDS:
        assert by_id[vid]["kind"] == "video"
        assert by_id[vid]["status"] == "indexed"
        assert by_id[vid]["title"]


@pytest.mark.mutating
def test_a_video_and_a_document_appear_side_by_side(
    client: httpx.Client, auth: dict, a_pending_paper: str
) -> None:
    """The point of the endpoint: one read, both kinds."""
    body = client.get("/admin/sources", headers=auth).json()
    by_id = {s["id"]: s for s in body["sources"]}

    assert by_id[CORPUS_IDS[0]]["kind"] == "video"

    doc = by_id[a_pending_paper]
    assert doc["kind"] == "paper"
    assert doc["status"] == "pending"
    assert doc["title"] == "RAG Survey"
    assert doc["uri"] == PAPER_URI
    assert doc["pct"] == 0


# ── Filters ───────────────────────────────────────────────────────────────────

@pytest.mark.mutating
def test_filter_by_kind(client: httpx.Client, auth: dict, a_pending_paper: str) -> None:
    body = client.get("/admin/sources", params={"kind": "paper"}, headers=auth).json()
    assert {s["kind"] for s in body["sources"]} == {"paper"}
    assert a_pending_paper in {s["id"] for s in body["sources"]}


def test_filter_by_status(client: httpx.Client, auth: dict) -> None:
    body = client.get("/admin/sources", params={"status": "indexed"},
                      headers=auth).json()
    assert {s["status"] for s in body["sources"]} == {"indexed"}


def test_an_unknown_kind_filter_is_400(client: httpx.Client, auth: dict) -> None:
    r = client.get("/admin/sources", params={"kind": "audio"}, headers=auth)
    assert r.status_code == 400


def test_an_unknown_status_filter_is_400(client: httpx.Client, auth: dict) -> None:
    r = client.get("/admin/sources", params={"status": "elsewhere"}, headers=auth)
    assert r.status_code == 400


# ── Paging, because the harness polls this ────────────────────────────────────

def test_paging_is_bounded_and_reports_total(client: httpx.Client, auth: dict) -> None:
    body = client.get("/admin/sources", params={"limit": 2}, headers=auth).json()
    assert len(body["sources"]) <= 2
    assert body["total"] >= len(body["sources"])
    assert body["limit"] == 2
    assert body["offset"] == 0


def test_an_oversized_limit_is_rejected(client: httpx.Client, auth: dict) -> None:
    assert client.get("/admin/sources", params={"limit": 10_000},
                      headers=auth).status_code == 422


def test_paging_covers_every_row_exactly_once(client: httpx.Client, auth: dict) -> None:
    """Ordering must be stable across pages. list_videos() sorts on created_at
    alone, which ties when a backfill inserts many rows in one clock tick — a
    row could then appear twice or never. list_sources() breaks the tie on id.
    """
    total = client.get("/admin/sources", headers=auth).json()["total"]
    seen: list[str] = []
    offset = 0
    while True:
        body = client.get("/admin/sources", params={"limit": 1, "offset": offset},
                          headers=auth).json()
        seen += [s["id"] for s in body["sources"]]
        if body["next_offset"] is None:
            break
        offset = body["next_offset"]
    assert len(seen) == total
    assert len(set(seen)) == total, "a row was paged twice"
