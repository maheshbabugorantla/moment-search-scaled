"""The video contract — what every later PR must not break.

"The provided video endpoints and the UI still work unmodified" is a
Definition-of-Done box and a rubric red line. Multi-source ingestion touches the
manifest schema, the ingest flow, retrieval and fusion; each of those can break
video without anyone noticing until the demo. This suite is the guard, written
against the baseline before `kind` exists.

Two notes on what is *not* asserted here, because the assignment brief and the
shipped app disagree and the app wins:

* There is no SSE stream. `POST /api/ask` (src/api/search.py) returns one JSON
  body; no `/ask_stream` route exists. There are no event names to pin.
* Citations carry no speaker/diarization field. The transcript branch is
  yt-dlp subtitles, which have no speaker labels. The observed payload is
  asserted below in full, and speaker is absent from it.

Safety rule for anything added here: never POST a corpus video URL to
/api/videos. See the note in conftest.CORPUS_IDS.
"""
from __future__ import annotations

import re

import httpx
import pytest

from conftest import CORPUS_IDS

# A well-formed but nonexistent YouTube id (11 chars, matches the app's regex).
# Registering it exercises the real register path without touching the corpus:
# the purge in t_embed_index only ever targets this id's own (empty) point set.
THROWAWAY_URL = "https://youtu.be/zzTESTzz001"
THROWAWAY_ID = "yt_zzTESTzz001"

TIMESTAMP_RE = re.compile(r"^\d{2,}:\d{2}$")


# ── The UI still serves ───────────────────────────────────────────────────────

def test_root_serves_the_search_ui(client: httpx.Client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()


def test_health_is_200(client: httpx.Client) -> None:
    assert client.get("/api/health").status_code == 200


def test_config_is_json(client: httpx.Client) -> None:
    r = client.get("/api/config")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


# ── Auth is enforced on the write path ────────────────────────────────────────

def test_register_without_token_is_401(client: httpx.Client) -> None:
    r = client.post("/api/videos", json={"url": THROWAWAY_URL})
    assert r.status_code == 401


def test_register_with_wrong_token_is_401(client: httpx.Client) -> None:
    r = client.post(
        "/api/videos",
        json={"url": THROWAWAY_URL},
        headers={"Authorization": "Bearer definitely-not-the-token"},
    )
    assert r.status_code == 401


# ── Register: shape-only, no ingest of anything real ──────────────────────────

def test_register_rejects_a_non_youtube_url(client: httpx.Client, auth: dict) -> None:
    r = client.post("/api/videos", json={"url": "https://example.com/nope"}, headers=auth)
    assert r.status_code == 400


def test_register_requires_url_or_upload_pair(client: httpx.Client, auth: dict) -> None:
    r = client.post("/api/videos", json={}, headers=auth)
    assert r.status_code == 400


@pytest.mark.mutating
def test_register_returns_202_and_the_documented_body(
    client: httpx.Client, auth: dict
) -> None:
    """The exact shape the UI polls on. 202 + {"video_id", "status"}.

    Note the key is `video_id`, not `id` — src/api/videos.py returns
    {"video_id": row["id"], "status": ...}. The manifest work in Epic 1 must
    preserve this spelling.
    """
    try:
        r = client.post("/api/videos", json={"url": THROWAWAY_URL}, headers=auth)
        assert r.status_code == 202
        body = r.json()
        assert body["video_id"] == THROWAWAY_ID
        assert body["status"] == "pending"
        # `flow_run_id` appears only in FIFO mode (ENABLE_FAIR_DISPATCH=false),
        # so it is optional — but nothing else may appear.
        assert set(body) <= {"video_id", "status", "flow_run_id"}
    finally:
        client.delete(f"/api/videos/{THROWAWAY_ID}", headers=auth)


# ── Status / lifecycle ────────────────────────────────────────────────────────

_PUBLIC_FIELDS = {
    "id", "source", "url", "title", "status", "error", "frame_count",
    "progress", "attempts", "created_at", "updated_at", "is_sample",
}


def test_list_videos_exposes_the_public_field_set(client: httpx.Client) -> None:
    r = client.get("/api/videos")
    assert r.status_code == 200
    videos = r.json()["videos"]
    assert videos, "no videos in the manifest — is the corpus indexed?"
    for v in videos:
        assert set(v) == _PUBLIC_FIELDS


def test_corpus_talks_are_indexed(client: httpx.Client) -> None:
    for vid in CORPUS_IDS:
        r = client.get(f"/api/videos/{vid}")
        assert r.status_code == 200, f"{vid} missing from the manifest"
        row = r.json()
        assert row["status"] == "indexed", f"{vid} is {row['status']}"
        assert row["frame_count"] > 0


def test_unknown_video_is_404(client: httpx.Client) -> None:
    assert client.get("/api/videos/yt_doesNotExis").status_code == 404


def test_corpus_talks_are_delete_protected(client: httpx.Client, auth: dict) -> None:
    """Sample ids are a hardcoded frozenset (src/samples.py) and the delete
    endpoint refuses them before touching storage. This both pins the behaviour
    and is why the suite cannot accidentally destroy the corpus."""
    for vid in CORPUS_IDS:
        assert client.delete(f"/api/videos/{vid}", headers=auth).status_code == 403


# ── Ask: the answer and its citations ─────────────────────────────────────────

_CITATION_FIELDS = {
    "n", "video_id", "title", "url", "source", "ms", "timestamp", "idx",
    "thumbnail", "media_url", "deeplink", "score", "transcript", "modalities",
}

# Added by Epic 4 (REC-314). Listed separately from the fields above so this
# test still states, precisely, which keys the provided video UI depended on
# before multi-source existed.
_EPIC4_FIELDS = {"sourceId", "kind", "locator", "label"}


@pytest.fixture(scope="module")
def answer(client: httpx.Client) -> dict:
    """Deliberately scoped to the video corpus.

    Before Epic 4 this question could only be answered by a video, so an
    unscoped ask WAS a video ask. Papers and posts about attention are now in
    the index and legitimately outrank some talks — so the video contract is
    pinned under the explicit `video_ids` scope the UI already sends, which is
    the only way to keep asserting the original guarantee rather than a
    weakened version of it. Cross-source behaviour is tested separately, in
    tests/test_cross_source_contract.py.
    """
    r = client.post(
        "/api/ask",
        json={"question": "what is attention in transformers?", "top_k": 5,
              "video_ids": sorted(CORPUS_IDS)},
    )
    assert r.status_code == 200
    return r.json()


def test_ask_returns_an_answer_and_citations(answer: dict) -> None:
    assert set(answer) >= {"question", "citations", "answer"}
    assert answer["answer"].strip()
    assert answer["citations"], "no citations — retrieval returned nothing"


def test_every_citation_carries_the_ui_contract(answer: dict) -> None:
    """The UI reads ms/timestamp/deeplink/thumbnail off each citation to seek
    the player. Epic 4 adds `kind` and a locator; these fields must survive."""
    for c in answer["citations"]:
        assert set(c) == _CITATION_FIELDS | _EPIC4_FIELDS
        assert c["kind"] == "video", "video_ids scope leaked another kind"
        assert isinstance(c["ms"], int) and c["ms"] >= 0
        assert TIMESTAMP_RE.match(c["timestamp"]), c["timestamp"]
        assert c["video_id"] in CORPUS_IDS
        assert isinstance(c["score"], (int, float))
        assert c["modalities"], "a citation with no modality is unexplainable"


def test_a_video_citation_locator_agrees_with_its_flat_timestamp(answer: dict) -> None:
    """`ms` and `locator.start_ms` are the same number reached two ways. They
    are allowed to coexist (the UI reads the flat one) but never to disagree —
    a divergence would mean the player seeks somewhere the citation doesn't
    claim."""
    for c in answer["citations"]:
        assert c["locator"]["start_ms"] == c["ms"]
        assert c["locator"]["end_ms"] > c["ms"]
        assert c["sourceId"] == c["video_id"]


def test_citation_deeplink_seeks_to_the_cited_moment(answer: dict) -> None:
    for c in answer["citations"]:
        assert f"t={c['ms'] // 1000}" in c["deeplink"]


def test_citations_are_ranked_best_first(answer: dict) -> None:
    scores = [c["score"] for c in answer["citations"]]
    assert scores == sorted(scores, reverse=True)


def test_empty_question_is_400(client: httpx.Client) -> None:
    assert client.post("/api/ask", json={"question": "   "}).status_code == 400


def test_ask_scoped_to_one_video_only_cites_that_video(client: httpx.Client) -> None:
    """The multi-select scope the UI sends. Epic 4's cross-source blending must
    keep honouring it."""
    target = "yt_wjZofJX0v4M"
    r = client.post(
        "/api/ask",
        json={"question": "what is attention?", "video_ids": [target], "top_k": 5},
    )
    assert r.status_code == 200
    cites = r.json()["citations"]
    assert cites, "scoped ask returned nothing"
    assert {c["video_id"] for c in cites} == {target}
