"""The document admin contract — POST /admin/documents.

The async promise, provable before a parser exists: validate the shape, insert a
`pending` row, return 202. Nothing here reaches the network, which is the point
— see the note in src/api/admin.py about the SLA probe.

Every test cleans up the row it creates, so the manifest is left as it was found.
"""
from __future__ import annotations

import time

import httpx
import pytest

from conftest import CORPUS_IDS

PAPER_URI = "https://example.com/contract-test-paper.pdf"
DECK_URI = "storage://decks/contract-test-deck.pdf"
POST_URI = "https://example.com/contract-test-post.md"
# Deliberately unreachable — a 202 here is the assertion that matters.
UNREACHABLE_URI = "https://example.invalid/probe_7.pdf"


@pytest.fixture
def registered(client: httpx.Client, auth: dict):
    """Registers documents and removes them afterwards, whatever the test does."""
    created: list[str] = []

    def _register(uri: str, kind: str, title: str | None = None) -> httpx.Response:
        body = {"uri": uri, "kind": kind}
        if title is not None:
            body["title"] = title
        r = client.post("/admin/documents", json=body, headers=auth)
        if r.status_code == 202:
            created.append(r.json()["id"])
        return r

    yield _register

    for doc_id in created:
        client.delete(f"/api/videos/{doc_id}", headers=auth)


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_register_document_without_token_is_401(client: httpx.Client) -> None:
    r = client.post("/admin/documents", json={"uri": PAPER_URI, "kind": "paper"})
    assert r.status_code == 401


def test_register_document_with_wrong_token_is_401(client: httpx.Client) -> None:
    r = client.post(
        "/admin/documents",
        json={"uri": PAPER_URI, "kind": "paper"},
        headers={"Authorization": "Bearer nope"},
    )
    assert r.status_code == 401


# ── The documented request/response ───────────────────────────────────────────

@pytest.mark.mutating
def test_paper_returns_202_and_the_documented_body(registered) -> None:
    r = registered(PAPER_URI, "paper", "RAG Survey")
    assert r.status_code == 202
    body = r.json()
    assert set(body) == {"id", "status", "kind"}
    assert body["id"].startswith("doc_")
    assert body["status"] == "pending"
    assert body["kind"] == "paper"


@pytest.mark.mutating
def test_deck_returns_202_with_a_storage_uri(registered) -> None:
    r = registered(DECK_URI, "deck", "KDD Keynote")
    assert r.status_code == 202
    assert r.json()["kind"] == "deck"


@pytest.mark.mutating
def test_post_returns_202_with_a_markdown_uri(registered) -> None:
    """The schema gate (REC-334): before migration 003 this died at the INSERT
    and surfaced as a 502 — the wrong error for a supported kind."""
    r = registered(POST_URI, "post", "Scaling laws, revisited")
    assert r.status_code == 202
    assert r.json()["kind"] == "post"


@pytest.mark.mutating
def test_the_row_is_visible_as_pending(client: httpx.Client, registered) -> None:
    """A deck, deliberately: decks have no flow, so `pending` is stable. A
    paper would be claimed by the dispatcher within one ~3s tick and could be
    `queued`/`fetching` by the time this reads it back."""
    doc_id = registered(DECK_URI, "deck", "KDD Keynote").json()["id"]
    row = client.get(f"/api/videos/{doc_id}").json()
    assert row["status"] == "pending"
    assert row["title"] == "KDD Keynote"


@pytest.mark.mutating
def test_the_same_uri_is_idempotent(registered) -> None:
    """Re-posting a URI updates one row rather than accumulating duplicates —
    the property `yt_<id>` gives videos."""
    first = registered(PAPER_URI, "paper").json()["id"]
    second = registered(PAPER_URI, "paper").json()["id"]
    assert first == second


# ── Shape-only validation ─────────────────────────────────────────────────────

@pytest.mark.mutating
def test_an_unreachable_uri_still_returns_202(registered) -> None:
    """The SLA probe posts ~30 documents at fake URIs and only samples
    accept-latency on a 202. Validating reachability here would read as an
    infinite latency in the gate weeks later."""
    assert registered(UNREACHABLE_URI, "paper").status_code == 202


@pytest.mark.parametrize("kind", ["video", "audio", "", "PAPER"])
def test_an_unknown_kind_is_400(client: httpx.Client, auth: dict, kind: str) -> None:
    """`video` is rejected here too — POST /api/videos owns that kind."""
    r = client.post("/admin/documents", json={"uri": PAPER_URI, "kind": kind},
                    headers=auth)
    assert r.status_code == 400


@pytest.mark.parametrize("uri", ["", "   ", "ftp://example.com/x.pdf", "not-a-uri"])
def test_a_bad_uri_is_400(client: httpx.Client, auth: dict, uri: str) -> None:
    r = client.post("/admin/documents", json={"uri": uri, "kind": "paper"},
                    headers=auth)
    assert r.status_code == 400


def test_a_missing_field_is_422(client: httpx.Client, auth: dict) -> None:
    """Pydantic's own validation — a body with no kind never reaches _validate."""
    r = client.post("/admin/documents", json={"uri": PAPER_URI}, headers=auth)
    assert r.status_code == 422


# ── The async promise ─────────────────────────────────────────────────────────

def _median_ms(fn, n: int = 5) -> float:
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return sorted(samples)[len(samples) // 2]


@pytest.mark.mutating
def test_accept_does_not_block_on_work(
    client: httpx.Client, auth: dict, registered
) -> None:
    """Accept must return before any fetching or parsing happens.

    Asserted relative to a single manifest read rather than against an absolute
    millisecond bar. The DB is managed Postgres over the internet, so one
    round-trip is ~300ms here and that floor dominates everything: a read-only
    `GET /api/videos/{id}` costs the same as this endpoint's one INSERT. An
    absolute 300ms bar would fail on a correct implementation and pass on a
    LAN — it measures the network, not the contract.

    What the contract actually promises is that accept does no work. If someone
    adds a fetch or a parse to this path, it stops being one round-trip and this
    ratio blows out.
    """
    corpus_id = CORPUS_IDS[0]
    baseline = _median_ms(lambda: client.get(f"/api/videos/{corpus_id}"))
    accept = _median_ms(
        lambda: registered(f"https://example.invalid/probe_{time.time_ns()}.pdf",
                           "paper"))
    assert accept < baseline * 3 + 150, (
        f"accept {accept:.0f}ms vs a single manifest read {baseline:.0f}ms — "
        "the accept path looks like it is doing real work")
    # Backstop: however slow the DB is, accept is never seconds.
    assert accept < 2000, f"accept took {accept:.0f}ms"


@pytest.mark.mutating
def test_a_deck_is_not_claimed_by_the_dispatcher(
    client: httpx.Client, registered
) -> None:
    """The original guarantee, now scoped to kinds WITHOUT a flow. wfq_claim()
    used to select any pending row and hand it to jobs.enqueue_video(), pushing
    a document through yt-dlp. Papers have their own flow now (REC-309) and ARE
    claimed; a deck still is not — it must sit `pending` until ingest_deck
    exists, not be crashed through a pipeline that can't handle it.
    """
    doc_id = registered(DECK_URI, "deck").json()["id"]
    # The dispatcher polls on an interval; give it several rounds to misbehave.
    time.sleep(12)
    assert client.get(f"/api/videos/{doc_id}").json()["status"] == "pending"


@pytest.mark.mutating
def test_a_post_with_a_dead_uri_is_dispatched_and_fails_readably(
    client: httpx.Client, registered
) -> None:
    """Posts have a flow now (REC-336), so they ARE claimed — and a
    deterministic fetch failure (this URI 404s or serves HTML, never markdown)
    reaches `failed` fast, because SourceError skips the 30s+120s ladder."""
    doc_id = registered(POST_URI, "post").json()["id"]
    deadline = time.monotonic() + 60
    row = {}
    while time.monotonic() < deadline:
        row = client.get(f"/api/videos/{doc_id}").json()
        if row.get("status") == "failed":
            break
        time.sleep(2)
    assert row.get("status") == "failed", (
        f"post stuck in {row.get('status')!r} — was it never dispatched, "
        "or did a dead URI burn the full retry ladder?")
    assert row.get("error"), "failed with no reason recorded"
    assert "Traceback" not in row["error"]
    assert "\n" not in row["error"].strip()


@pytest.mark.mutating
def test_a_paper_with_a_dead_uri_is_dispatched_and_fails_readably(
    client: httpx.Client, registered
) -> None:
    """The flip side of the deck test: a registered paper IS claimed now, and a
    deterministic fetch failure (this URI 404s or serves HTML, never a PDF)
    reaches `failed` fast — PaperSourceError skips the 30s+120s retry ladder —
    with a one-line human-readable reason, not a traceback.
    """
    doc_id = registered(PAPER_URI, "paper").json()["id"]
    deadline = time.monotonic() + 60
    row = {}
    while time.monotonic() < deadline:
        row = client.get(f"/api/videos/{doc_id}").json()
        if row.get("status") == "failed":
            break
        time.sleep(2)
    assert row.get("status") == "failed", (
        f"paper stuck in {row.get('status')!r} — was it never dispatched, "
        "or did a dead URI burn the full retry ladder?")
    assert row.get("error"), "failed with no reason recorded"
    assert "Traceback" not in row["error"]
    assert "\n" not in row["error"].strip()
