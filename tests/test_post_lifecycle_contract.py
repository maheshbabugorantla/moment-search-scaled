"""End-to-end post ingest — the REC-336 verify, black-box over HTTP.

Registers the API's own fixture markdown (GET /admin/fixtures/tiny.md,
generated in memory) and watches the row walk the real lifecycle to `indexed`.
The fixture URI must be resolvable BY THE WORKER, which lives on the compose
network — hence the http://api:8000 default rather than this suite's own
CONTRACT_BASE_URL.

What is deliberately NOT asserted here: that a query returns a post. Posts are
excluded from search_text() until Epic 4 can render an anchor citation, so a
search-based assertion would only pass by weakening the guard. The indexed
units are checked directly instead.
"""
from __future__ import annotations

import os
import time

import httpx
import pytest

WORKER_BASE_URL = os.getenv("CONTRACT_WORKER_BASE_URL", "http://api:8000")
FIXTURE_URI = f"{WORKER_BASE_URL}/admin/fixtures/tiny.md"

# The same vocabulary a paper walks — posts add no status of their own, which
# is itself the assertion.
POST_STATUSES = {"pending", "queued", "fetching", "parsing", "chunking",
                 "embedding", "indexed", "skipped", "failed"}

INDEXED_DEADLINE_S = 180


@pytest.fixture
def registered_post(client: httpx.Client, auth: dict):
    r = client.post("/admin/documents",
                    json={"uri": FIXTURE_URI, "kind": "post",
                          "title": "Fixture Post"},
                    headers=auth)
    assert r.status_code == 202
    doc_id = r.json()["id"]
    yield doc_id
    client.delete(f"/api/videos/{doc_id}", headers=auth)


def _source_row(client: httpx.Client, auth: dict, doc_id: str) -> dict:
    body = client.get("/admin/sources", params={"kind": "post", "limit": 500},
                      headers=auth).json()
    return next((s for s in body["sources"] if s["id"] == doc_id), {})


def _wait_for_terminal(client: httpx.Client, auth: dict, doc_id: str,
                       deadline_s: float = INDEXED_DEADLINE_S,
                       ) -> tuple[dict, list[str], list[int]]:
    statuses: list[str] = []
    pcts: list[int] = []
    row: dict = {}
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        row = _source_row(client, auth, doc_id)
        status = row.get("status")
        pcts.append(row.get("pct") or 0)
        if not statuses or statuses[-1] != status:
            statuses.append(status)
        if status in ("indexed", "failed", "skipped"):
            return row, statuses, pcts
        time.sleep(2)
    return row, statuses, pcts


# ── The fixture itself ────────────────────────────────────────────────────────

def test_the_fixture_markdown_is_served(client: httpx.Client) -> None:
    r = client.get("/admin/fixtures/tiny.md")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    # Opens with prose, NOT a heading — that is what makes `_top` reachable.
    assert not r.text.lstrip().startswith("#")
    assert "\n# " in r.text and "\n## " in r.text


def test_the_fixture_exercises_a_duplicate_heading(client: httpx.Client) -> None:
    """The `-1` suffix path only gets covered end to end if the fixture
    actually repeats a heading — assert the fixture, not just the parser."""
    body = client.get("/admin/fixtures/tiny.md").text
    headings = [ln for ln in body.splitlines() if ln.startswith("## ")]
    assert len(headings) != len(set(headings)), "fixture no longer repeats a heading"


# ── The lifecycle walk ────────────────────────────────────────────────────────

@pytest.mark.mutating
def test_a_real_post_reaches_indexed_with_monotone_pct(
    client: httpx.Client, auth: dict, registered_post: str
) -> None:
    """The whole promise in one walk: pending -> ... -> indexed with nobody
    touching anything after the 202, every observed status inside the vocabulary
    a PAPER already defined (a post adds no status of its own), pct never going
    backwards, and pct == 100 exactly when the status reads `indexed`."""
    row, statuses, pcts = _wait_for_terminal(client, auth, registered_post)
    assert set(statuses) <= POST_STATUSES, f"undocumented status in {statuses}"
    assert row.get("status") == "indexed", (
        f"terminal status {row.get('status')!r} (error={row.get('error')!r}); "
        f"observed walk: {statuses}")
    assert pcts == sorted(pcts), f"pct went backwards: {pcts}"
    assert row["pct"] == 100

    assert client.get(f"/api/videos/{registered_post}").json()["frame_count"] > 0


@pytest.mark.mutating
def test_reingesting_the_same_uri_is_idempotent(
    client: httpx.Client, auth: dict, registered_post: str
) -> None:
    """Deterministic anchors and chunk ids: the same markdown re-registered
    overwrites its previous points, so the indexed unit count is identical."""
    first, _, _ = _wait_for_terminal(client, auth, registered_post)
    assert first.get("status") == "indexed", first.get("error")
    count = client.get(f"/api/videos/{registered_post}").json()["frame_count"]

    r = client.post("/admin/documents",
                    json={"uri": FIXTURE_URI, "kind": "post"}, headers=auth)
    assert r.status_code == 202
    assert r.json()["id"] == registered_post  # same URI -> same row

    second, _, _ = _wait_for_terminal(client, auth, registered_post)
    assert second.get("status") == "indexed", second.get("error")
    assert client.get(f"/api/videos/{registered_post}").json()["frame_count"] == count


@pytest.mark.mutating
def test_the_indexed_points_carry_kind_post_and_plausible_anchors(
    client: httpx.Client, auth: dict, registered_post: str
) -> None:
    """What the citation will later read. Asserted against Qdrant directly
    because the query path deliberately excludes posts until Epic 4 — reaching
    for a search assertion here would mean weakening that guard to make a test
    pass.

    Two anchors matter: `_top` (content before the first heading resolves on
    every renderer) and `retrieval-1` (the fixture repeats a heading, and the
    FIRST occurrence must keep the bare slug)."""
    row, _, _ = _wait_for_terminal(client, auth, registered_post)
    assert row.get("status") == "indexed", row.get("error")

    from src.rag import vector_store
    from src.config import TEXT_COLLECTION

    points, _ = vector_store.client().scroll(
        collection_name=TEXT_COLLECTION,
        scroll_filter=vector_store._user_filter("default", registered_post),
        limit=200, with_payload=True)
    assert points, "indexed but no points in the text collection"

    payloads = [p.payload for p in points]
    assert {p["kind"] for p in payloads} == {"post"}
    assert all(p["modality"] == "text" for p in payloads)
    anchors = {p["anchor"] for p in payloads}
    assert "_top" in anchors
    assert "retrieval" in anchors and "retrieval-1" in anchors, (
        f"duplicate-heading numbering wrong: {sorted(anchors)}")
    # The heading path rides along for citation display, and every payload
    # states whether its anchor can actually scroll the live post.
    assert all(isinstance(p["anchor_native"], bool) for p in payloads)
    assert all(p["text"].strip() for p in payloads)


@pytest.mark.mutating
def test_the_decorative_image_is_never_indexed(
    client: httpx.Client, auth: dict, registered_post: str
) -> None:
    """The requirement REC-337 actually states, asserted from the outside: a
    decorative image must never become citable.

    The fixture's banner is 1200x100 — dropped by a shape rule (its 100px
    height trips the minimum-dimension check before the aspect check is
    reached) without the classifier being consulted at all, so this assertion
    is deterministic and does not depend on what CLIP makes of a generated
    PNG. The 600x400 chart clears every heuristic and scores 0.29 informative
    against a 0.22 floor, so the keep path is live rather than vacuous — and
    any surviving image must carry its verdict, keeping a future tuning
    change auditable.
    """
    row, _, _ = _wait_for_terminal(client, auth, registered_post)
    assert row.get("status") == "indexed", row.get("error")

    from src.config import QDRANT_COLLECTION
    from src.rag import vector_store

    points, _ = vector_store.client().scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=vector_store._user_filter("default", registered_post),
        limit=100, with_payload=True)

    frames = [p.payload for p in points]
    assert len(frames) <= 1, (
        "the 1200x100 banner reached the index — the aspect rule did not fire")
    for f in frames:
        # .get, not [] — a scoping bug should surface as a readable assertion
        # here, not as a KeyError against a video frame that carries no `kind`.
        assert f.get("kind") == "post"
        assert f.get("modality") == "frame"
        assert f.get("img_class") == "informative"
        assert isinstance(f.get("img_score"), float)
        assert f.get("anchor"), "an indexed image with no anchor cannot be cited"


# ── The query path must survive an indexed post ──────────────────────────────

@pytest.mark.mutating
def test_asking_a_question_still_works_with_a_post_indexed(
    client: httpx.Client, auth: dict, registered_post: str
) -> None:
    """The gap that let a 500 ship: every other test here reads Qdrant
    directly, so nothing exercised /api/ask while a post was in the index.

    A post's kept images live in the CLIP collection — they reuse the
    frame_key layout on purpose — but they carry an `anchor`, not an `ms`.
    search_text() excluded documents; search() did not, so a post image
    ranking in the visual top-K reached the citation builder, which reads
    fr["ms"] unconditionally and blew up the whole answer.

    The question deliberately targets the fixture post's own subject matter,
    so its chunks and its chart are the most likely things to rank.
    """
    row, _, _ = _wait_for_terminal(client, auth, registered_post)
    assert row.get("status") == "indexed", row.get("error")

    r = client.post("/api/ask",
                    json={"question": "How does reciprocal rank fusion merge "
                                      "ranked lists from different scorers?"})
    assert r.status_code == 200, (
        f"/api/ask returned {r.status_code} with a post indexed — "
        f"a document leaked into a branch that assumes video: {r.text[:300]}")

    body = r.json()
    for c in body.get("citations", []):
        assert (c.get("kind") or "video") == "video", (
            f"a {c.get('kind')} citation reached the video Q&A path; it has no "
            "timestamp to seek to and its deeplink cannot be rendered yet")
        assert c.get("ms") is not None, f"citation with no timestamp: {c}"


# ── The flowless-kind guarantee still holds ──────────────────────────────────

def test_decks_remain_the_only_kind_without_a_flow(
    client: httpx.Client, auth: dict
) -> None:
    """Every kind the dispatcher claims must have somewhere to send it. This
    reads the config the dispatcher reads, so adding a fifth kind to
    DISPATCH_KINDS without a route fails here rather than in production."""
    from src.config import DISPATCH_KINDS, SOURCE_KINDS

    assert set(DISPATCH_KINDS) == set(SOURCE_KINDS) - {"deck"}
