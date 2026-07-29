"""Deterministic fetch failures skip the retry ladder — both flows.

Found in production data rather than by reasoning: a 12-hour sweep of Prefect
runs showed every video-flow failure was `ValueError: no manifest row for
yt_zzTESTzz001` — a row the contract suite deletes between dispatch and
pickup. The paper flow already classified that case as deterministic; the
video flow did not, so each occurrence held one of two worker slots for the
full 30s + 120s backoff before failing exactly as it had the first time.

These tests pin the classification, not the plumbing: they call the tasks'
underlying functions directly, so no Prefect engine or database is involved.
"""
from __future__ import annotations

import pytest

from src.ingest import paper, pipeline
from src.ingest.errors import SourceError, retry_unless_source_error


class _State:
    """The slice of a Prefect state the retry condition touches."""

    def __init__(self, exc: BaseException):
        self._exc = exc

    def result(self):
        raise self._exc


# ── The classification ────────────────────────────────────────────────────────

def test_a_source_error_is_not_retried() -> None:
    assert retry_unless_source_error(None, None, _State(SourceError("404"))) is False


def test_a_transient_error_is_retried() -> None:
    assert retry_unless_source_error(None, None, _State(TimeoutError("slow"))) is True


def test_the_paper_alias_still_classifies() -> None:
    """PaperSourceError is an alias now; every existing raise site must keep
    hitting the same branch."""
    assert paper.PaperSourceError is SourceError
    err = paper.PaperSourceError("not a PDF")
    assert retry_unless_source_error(None, None, _State(err)) is False


# ── Both fetch tasks classify a vanished row the same way ────────────────────

def test_video_fetch_raises_a_source_error_for_a_vanished_row(monkeypatch) -> None:
    monkeypatch.setattr(pipeline.db, "set_status", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.db, "get_video", lambda vid: None)
    with pytest.raises(SourceError, match="no manifest row"):
        pipeline.t_fetch.fn("yt_gone", "default")


def test_paper_fetch_raises_a_source_error_for_a_vanished_row(monkeypatch) -> None:
    from src.ingest import paper_pipeline

    monkeypatch.setattr(paper_pipeline.db, "set_status", lambda *a, **k: None)
    monkeypatch.setattr(paper_pipeline.db, "get_video", lambda vid: None)
    with pytest.raises(SourceError, match="no manifest row"):
        paper_pipeline.t_fetch_paper.fn("doc_gone", "default")


def test_both_fetch_tasks_carry_the_retry_condition() -> None:
    """The classification is inert unless the task consults it."""
    from src.ingest import paper_pipeline

    assert pipeline.t_fetch.retry_condition_fn is retry_unless_source_error
    assert paper_pipeline.t_fetch_paper.retry_condition_fn is retry_unless_source_error
