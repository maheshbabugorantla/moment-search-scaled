"""End-to-end paper ingest — the REC-309 verify, black-box over HTTP.

Registers the API's own fixture PDF (GET /admin/fixtures/tiny.pdf — generated
in memory, so no external network and no binary in git) and watches the row
walk the real lifecycle to `indexed`. The fixture URI must be resolvable BY THE
WORKER, which lives on the compose network — hence the http://api:8000 default
rather than this suite's own CONTRACT_BASE_URL.

Timing: the dispatcher ticks every ~3s and the whole flow on a 3-page PDF is
seconds of work; the generous deadline below absorbs a cold bge model load.
"""
from __future__ import annotations

import os
import time

import httpx
import pytest

# Where the WORKER can reach the API (compose service DNS), not where pytest can.
WORKER_BASE_URL = os.getenv("CONTRACT_WORKER_BASE_URL", "http://api:8000")
FIXTURE_URI = f"{WORKER_BASE_URL}/admin/fixtures/tiny.pdf"

# The documented lifecycle vocabulary — a status outside this set is a bug
# whatever the outcome of the run.
PAPER_STATUSES = {"pending", "queued", "fetching", "parsing", "chunking",
                  "embedding", "indexed", "skipped", "failed"}

INDEXED_DEADLINE_S = 180


@pytest.fixture
def registered_paper(client: httpx.Client, auth: dict):
    r = client.post("/admin/documents",
                    json={"uri": FIXTURE_URI, "kind": "paper",
                          "title": "Fixture Paper"},
                    headers=auth)
    assert r.status_code == 202
    doc_id = r.json()["id"]
    yield doc_id
    client.delete(f"/api/videos/{doc_id}", headers=auth)


def _source_row(client: httpx.Client, auth: dict, doc_id: str) -> dict:
    body = client.get("/admin/sources", params={"kind": "paper", "limit": 500},
                      headers=auth).json()
    return next((s for s in body["sources"] if s["id"] == doc_id), {})


def _wait_for_terminal(client: httpx.Client, auth: dict, doc_id: str,
                       deadline_s: float = INDEXED_DEADLINE_S,
                       ) -> tuple[dict, list[str], list[int]]:
    """Poll /admin/sources until a terminal status, recording every status and
    pct observed — the same read the resilience harness and the UI poll."""
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


def test_the_fixture_pdf_is_served(client: httpx.Client) -> None:
    r = client.get("/admin/fixtures/tiny.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF-")


@pytest.mark.mutating
def test_a_real_pdf_reaches_indexed_with_monotone_pct(
    client: httpx.Client, auth: dict, registered_paper: str
) -> None:
    """The whole promise in one walk: pending -> ... -> indexed without a human
    touching anything after the 202, every observed status inside the
    documented vocabulary, pct never going backwards, and pct == 100 exactly
    when the status reads `indexed` (REC-310)."""
    row, statuses, pcts = _wait_for_terminal(client, auth, registered_paper)
    assert set(statuses) <= PAPER_STATUSES, f"undocumented status in {statuses}"
    assert row.get("status") == "indexed", (
        f"terminal status {row.get('status')!r} (error={row.get('error')!r}); "
        f"observed walk: {statuses}")
    assert pcts == sorted(pcts), f"pct went backwards: {pcts}"
    assert row["pct"] == 100

    # The count of indexed units (chunks) rides in frame_count on the video
    # surface — the one read the delete/status endpoints already expose.
    assert client.get(f"/api/videos/{registered_paper}").json()["frame_count"] > 0


@pytest.mark.mutating
def test_reingesting_the_same_uri_is_idempotent(
    client: httpx.Client, auth: dict, registered_paper: str
) -> None:
    """Deterministic point ids: a re-registered paper overwrites its previous
    points, so the indexed unit count is identical run over run."""
    first, _, _ = _wait_for_terminal(client, auth, registered_paper)
    assert first.get("status") == "indexed", first.get("error")
    count = client.get(f"/api/videos/{registered_paper}").json()["frame_count"]

    r = client.post("/admin/documents",
                    json={"uri": FIXTURE_URI, "kind": "paper"}, headers=auth)
    assert r.status_code == 202
    assert r.json()["id"] == registered_paper  # same URI -> same row

    second, _, _ = _wait_for_terminal(client, auth, registered_paper)
    assert second.get("status") == "indexed", second.get("error")
    assert client.get(f"/api/videos/{registered_paper}").json()["frame_count"] == count


def test_the_corpus_videos_report_pct_100(client: httpx.Client, auth: dict) -> None:
    """Migration 002's backfill: sources indexed before pct existed must read
    100, not 0 — otherwise every pre-REC-310 ingest stays visibly wrong."""
    body = client.get("/admin/sources", params={"kind": "video", "limit": 500},
                      headers=auth).json()
    indexed = [s for s in body["sources"] if s["status"] == "indexed"]
    assert indexed, "no indexed videos — is the corpus seeded?"
    for s in indexed:
        assert s["pct"] == 100, f"{s['id']} is indexed but reports pct {s['pct']}"
