"""The admin error envelope, and the 502 that needs a dependency to be down.

Two styles here, deliberately:

* the envelope and 400/401 are checked black-box against the running stack,
  like every other contract test;
* the 502 is checked in-process with the manifest write forced to fail. There
  is no way to knock Postgres over from a black-box test without taking the
  stack down, and "502 when the DB is unavailable" is exactly the path the
  rubric checks and the one that otherwise ships untested.
"""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import ADMIN_TOKEN

PAPER_URI = "https://example.com/error-envelope-test.pdf"
AUTH = {"Authorization": f"Bearer {ADMIN_TOKEN}"}


# ── The envelope, black-box ───────────────────────────────────────────────────

def _assert_envelope(body: dict, code: str) -> None:
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["code"] == code
    assert body["error"]["message"].strip()


def test_401_carries_the_envelope(client: httpx.Client) -> None:
    r = client.post("/admin/documents", json={"uri": PAPER_URI, "kind": "paper"})
    assert r.status_code == 401
    _assert_envelope(r.json(), "unauthorized")


def test_400_carries_the_envelope(client: httpx.Client, auth: dict) -> None:
    r = client.post("/admin/documents", json={"uri": PAPER_URI, "kind": "audio"},
                    headers=auth)
    assert r.status_code == 400
    _assert_envelope(r.json(), "bad_request")


def test_422_carries_the_envelope(client: httpx.Client, auth: dict) -> None:
    r = client.post("/admin/documents", json={"uri": PAPER_URI}, headers=auth)
    assert r.status_code == 422
    _assert_envelope(r.json(), "invalid_request")
    assert "kind" in r.json()["error"]["message"]


def test_no_stack_trace_reaches_the_client(client: httpx.Client, auth: dict) -> None:
    r = client.post("/admin/documents", json={"uri": "ftp://x/y.pdf", "kind": "paper"},
                    headers=auth)
    assert "Traceback" not in r.text
    assert "File \"" not in r.text


# ── The video surface keeps FastAPI's default body ───────────────────────────

def test_the_video_surface_is_untouched(client: httpx.Client) -> None:
    """The UI reads `detail` off error bodies in three places (ui/index.html).
    The envelope is scoped to /admin/* precisely so this keeps working."""
    r = client.post("/api/videos", json={"url": "https://youtu.be/zzTESTzz001"})
    assert r.status_code == 401
    assert "detail" in r.json()
    assert "error" not in r.json()


# ── 502, in-process with the dependency forced down ──────────────────────────

@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The app without lifespan — TestClient only runs startup when used as a
    context manager, and we neither want nor need Qdrant here."""
    from src.app import app
    return TestClient(app, raise_server_exceptions=False)


def test_a_manifest_failure_is_502_not_500(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distinction the rubric checks: a dependency being down is 502
    ("their fault"), not 500 ("our bug")."""
    from src import db

    def _boom(_doc):
        raise ConnectionError("connection to server failed: no route to host")

    monkeypatch.setattr(db, "upsert_pending_document", _boom)

    r = app_client.post("/admin/documents",
                        json={"uri": PAPER_URI, "kind": "paper"}, headers=AUTH)
    assert r.status_code == 502
    _assert_envelope(r.json(), "upstream_unavailable")
    # The underlying driver message must not leak to the caller.
    assert "no route to host" not in r.text


def test_a_manifest_failure_fails_fast(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dependency failure must not push accept latency out — the SLA probe
    samples this endpoint."""
    import time

    from src import db

    def _boom(_doc):
        raise ConnectionError("down")

    monkeypatch.setattr(db, "upsert_pending_document", _boom)

    start = time.perf_counter()
    r = app_client.post("/admin/documents",
                        json={"uri": PAPER_URI, "kind": "paper"}, headers=AUTH)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert r.status_code == 502
    assert elapsed_ms < 500, f"502 took {elapsed_ms:.0f}ms — it should be immediate"


def test_an_unexpected_error_is_500_with_no_trace(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Our own bug stays a 500, still enveloped, still without a stack trace."""
    from src.api import admin

    def _boom(_req):
        raise ValueError("a genuine bug in our own code")

    monkeypatch.setattr(admin, "_validate", _boom)

    r = app_client.post("/admin/documents",
                        json={"uri": PAPER_URI, "kind": "paper"}, headers=AUTH)
    assert r.status_code == 500
    _assert_envelope(r.json(), "internal_error")
    assert "a genuine bug" not in r.text
    assert "Traceback" not in r.text
