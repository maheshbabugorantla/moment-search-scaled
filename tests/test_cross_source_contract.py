"""Cross-source retrieval, black-box over HTTP — the Epic 4 payoff.

The epic's own done-when is "a single query emits citations carrying start_ms,
page and slide; a nonsense query returns zero citations; each locator jumps
correctly". This file asserts the parts a test can honestly assert.

What it deliberately does NOT do is pin a particular question to a particular
document. Which source wins a query is a property of the corpus, and a corpus
assertion fails for the wrong reason the moment someone indexes one more post.
The invariants below hold for ANY corpus: locators are well-formed for their
kind, no document reports a timestamp, no citation links somewhere it cannot
reach, and a question nothing answers is refused rather than answered badly.

Requires an indexed multi-kind corpus; skips cleanly when only videos exist,
so the file is honest on a fresh stack rather than green by vacuum.
"""
from __future__ import annotations

import httpx
import pytest

NONSENSE = "zxqv plorbnak wibbleflux zonktastic quibbernaut"


def _kinds_available(client: httpx.Client, auth: dict) -> set[str]:
    body = client.get("/admin/sources", params={"limit": 500}, headers=auth).json()
    return {s["kind"] for s in body["sources"] if s["status"] == "indexed"}


@pytest.fixture(scope="module")
def multikind(client: httpx.Client, auth: dict) -> set[str]:
    kinds = _kinds_available(client, auth)
    if len(kinds) < 2:
        pytest.skip(f"corpus has only {kinds} — nothing cross-source to assert")
    return kinds


def _ask(client: httpx.Client, q: str, **kw) -> dict:
    r = client.post("/api/ask", json={"question": q, **kw})
    assert r.status_code == 200, f"/api/ask {r.status_code}: {r.text[:300]}"
    return r.json()


# ── Every citation is well-formed FOR ITS KIND ───────────────────────────────

QUESTIONS = [
    "How should you choose what to work on?",
    "What is the economics of AI data centre capex?",
    "How does attention work in transformers?",
    "How do you evaluate an LLM application?",
]


@pytest.fixture(scope="module")
def answers(client: httpx.Client, multikind: set[str]) -> list[dict]:
    return [_ask(client, q) for q in QUESTIONS]


def test_no_query_across_the_corpus_errors(answers: list[dict]) -> None:
    """The regression that motivated the whole locator design: a document in
    the top-K used to reach a citation builder that read `ms` unconditionally
    and 500ed the answer. Asserted over several questions because it only
    fired when a document happened to rank."""
    assert len(answers) == len(QUESTIONS)


def test_every_citation_states_its_kind_and_a_locator(answers: list[dict]) -> None:
    for body in answers:
        for c in body["citations"]:
            assert c["kind"] in ("video", "paper", "deck", "post"), c["kind"]
            assert isinstance(c["locator"], dict) and c["locator"]
            assert c["label"], "a citation a reader cannot place is not a citation"
            assert c["sourceId"] == c["video_id"]


def test_a_locator_matches_the_shape_its_kind_promises(answers: list[dict]) -> None:
    for body in answers:
        for c in body["citations"]:
            loc, kind = c["locator"], c["kind"]
            if kind == "video":
                assert isinstance(loc["start_ms"], int)
                assert loc["end_ms"] > loc["start_ms"]
            elif kind == "paper":
                assert "page" in loc
            elif kind == "deck":
                assert "slide" in loc
            elif kind == "post":
                assert loc["anchor"], "a post citation with no anchor"
                assert isinstance(loc["anchor_native"], bool)


def test_only_a_video_reports_a_timestamp(answers: list[dict]) -> None:
    """The specific lie this epic exists to stop. A page has no time, and `0`
    is worse than `null` because a reader would act on it."""
    for body in answers:
        for c in body["citations"]:
            if c["kind"] == "video":
                assert isinstance(c["ms"], int) and c["timestamp"]
            else:
                assert c["ms"] is None and c["timestamp"] is None, \
                    f"{c['kind']} citation reported a timestamp: {c}"


def test_a_deeplink_is_either_reachable_or_absent(answers: list[dict]) -> None:
    """Never a compose-network hostname and never a storage:// key. A link that
    404s in the reader's browser is worse than an honest 'no link'."""
    for body in answers:
        for c in body["citations"]:
            link = c["deeplink"]
            if link is None:
                continue
            assert link.startswith(("http://", "https://", "/api/")), link
            for private in ("substack-fixtures", "posts-fixtures", "storage://"):
                assert private not in link, f"unreachable deeplink: {link}"


def test_a_post_deeplink_carries_its_anchor_when_the_anchor_resolves(
    answers: list[dict]
) -> None:
    """The reason posts are a kind at all: `{url}#{anchor}` scrolls the reader
    to the passage. A synthesised anchor has no `id=` in the rendered page, so
    it must NOT be appended — that fragment would silently do nothing."""
    seen = False
    for body in answers:
        for c in body["citations"]:
            if c["kind"] != "post" or not c["deeplink"]:
                continue
            loc = c["locator"]
            if loc["anchor_native"] and loc["anchor"] != "_top":
                assert c["deeplink"].endswith("#" + loc["anchor"]), c["deeplink"]
                seen = True
            else:
                assert "#" not in c["deeplink"].split("://", 1)[-1], (
                    f"a non-resolving anchor was appended anyway: {c['deeplink']}")
    if not seen:
        pytest.skip("no post citation with a native anchor ranked in these queries")


# ── Blending: one kind does not own every answer ─────────────────────────────

def test_at_least_one_question_is_answered_across_two_kinds(
    answers: list[dict], multikind: set[str]
) -> None:
    """REC-316's headline. Stated across the question set rather than per
    question: not every question SHOULD be cross-source — "how does attention
    work" being answered entirely by the two Karpathy talks is correct — but
    if no question in a deliberately broad set surfaces two kinds, blending
    is not working."""
    per_q = [{c["kind"] for c in b["citations"]} for b in answers]
    assert any(len(ks) >= 2 for ks in per_q), (
        f"every question came back single-kind: {per_q}")


def test_no_single_document_supplies_every_citation(answers: list[dict]) -> None:
    for body, q in zip(answers, QUESTIONS):
        cites = body["citations"]
        if len(cites) < 3:
            continue
        sources = {c["sourceId"] for c in cites}
        assert len(sources) > 1, f"{q!r} cited only {sources}"


# ── Abstention still works, and is honest about the corpus ───────────────────

def test_a_nonsense_question_does_not_invent_an_answer(
    client: httpx.Client, multikind: set[str]
) -> None:
    body = _ask(client, NONSENSE)
    assert body.get("abstained") is True, (
        f"answered nonsense with {len(body['citations'])} citations: "
        f"{body.get('answer', '')[:200]}")


def test_the_abstention_does_not_claim_the_corpus_is_only_video(
    client: httpx.Client, multikind: set[str]
) -> None:
    """A user whose papers and posts are indexed being told 'not in your
    videos' is being told something false about their own corpus."""
    body = _ask(client, NONSENSE)
    assert "your videos" not in (body.get("answer") or "")
