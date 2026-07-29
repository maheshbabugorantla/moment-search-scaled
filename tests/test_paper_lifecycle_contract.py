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


def _wait_for_terminal(client: httpx.Client, doc_id: str,
                       deadline_s: float = INDEXED_DEADLINE_S) -> tuple[dict, list[str]]:
    """Poll until a terminal status, recording every status observed."""
    seen: list[str] = []
    row: dict = {}
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        row = client.get(f"/api/videos/{doc_id}").json()
        status = row.get("status")
        if not seen or seen[-1] != status:
            seen.append(status)
        if status in ("indexed", "failed", "skipped"):
            return row, seen
        time.sleep(2)
    return row, seen


def test_the_fixture_pdf_is_served(client: httpx.Client) -> None:
    r = client.get("/admin/fixtures/tiny.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF-")


@pytest.mark.mutating
def test_a_real_pdf_reaches_indexed(
    client: httpx.Client, auth: dict, registered_paper: str
) -> None:
    """The whole promise in one walk: pending -> ... -> indexed, every observed
    status inside the documented vocabulary, and the terminal state reached
    without a human touching anything after the 202."""
    row, seen = _wait_for_terminal(client, registered_paper)
    assert set(seen) <= PAPER_STATUSES, f"undocumented status in {seen}"
    assert row.get("status") == "indexed", (
        f"terminal status {row.get('status')!r} (error={row.get('error')!r}); "
        f"observed walk: {seen}")
    # frame_count doubles as indexed-unit count — chunks, for a paper.
    assert row["frame_count"] > 0


@pytest.mark.mutating
def test_reingesting_the_same_uri_is_idempotent(
    client: httpx.Client, auth: dict, registered_paper: str
) -> None:
    """Deterministic point ids: a re-registered paper overwrites its previous
    points, so the indexed unit count is identical run over run."""
    first, _ = _wait_for_terminal(client, registered_paper)
    assert first.get("status") == "indexed", first.get("error")

    r = client.post("/admin/documents",
                    json={"uri": FIXTURE_URI, "kind": "paper"}, headers=auth)
    assert r.status_code == 202
    assert r.json()["id"] == registered_paper  # same URI -> same row

    second, _ = _wait_for_terminal(client, registered_paper)
    assert second.get("status") == "indexed", second.get("error")
    assert second["frame_count"] == first["frame_count"]
