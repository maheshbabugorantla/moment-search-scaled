"""Shared fixtures for the contract suite.

These are black-box tests: they talk to a running stack over HTTP rather than
importing the app. That is deliberate — the thing being guarded is the wire
contract the UI and the graders see, not the Python call signatures.

Run them against a stack that is already up (see tests/README.md).
"""
from __future__ import annotations

import os

import httpx
import pytest

# Inside the compose network the API answers to `api`; from the host it is
# localhost. Default to the host form and let compose override it.
BASE_URL = os.getenv("CONTRACT_BASE_URL", "http://localhost:8000")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# The four curated talks (src/samples.py). They are the read-only corpus these
# tests query. Nothing here may re-register them: re-ingesting a corpus video
# purges both Qdrant collections up front (src/ingest/pipeline.py) and a
# transcript failure is swallowed into "visual-only", so a bot-check during the
# test run would silently empty the text branch while still reporting `indexed`.
CORPUS_IDS = [
    "yt_LPZh9BOjkQs",
    "yt_wjZofJX0v4M",
    "yt_eMlx5fFNoYc",
    "yt_zjkBMFhNj_g",
]


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    with httpx.Client(base_url=BASE_URL, timeout=120.0) as c:
        yield c


@pytest.fixture(scope="session")
def auth() -> dict[str, str]:
    """Headers carrying the admin bearer token."""
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


@pytest.fixture(scope="session", autouse=True)
def _require_live_stack(client: httpx.Client) -> None:
    """Fail loudly and once if the stack isn't up, rather than 30 confusing
    connection errors."""
    try:
        r = client.get("/api/health")
    except httpx.HTTPError as exc:  # pragma: no cover - operator error path
        pytest.exit(f"No stack at {BASE_URL}: {exc}", returncode=2)
    if r.status_code != 200:
        pytest.exit(f"Stack at {BASE_URL} unhealthy: HTTP {r.status_code}", returncode=2)


@pytest.fixture(scope="session", autouse=True)
def _require_admin_token() -> None:
    """The 401 assertions are vacuous when ADMIN_TOKEN is unset, because
    require_auth() returns early as a dev convenience (src/api/videos.py). A
    green suite would then be meaningless, so refuse to run."""
    if not ADMIN_TOKEN:
        pytest.exit(
            "ADMIN_TOKEN is unset — auth is disabled and the 401 guards would "
            "pass without testing anything.",
            returncode=2,
        )
