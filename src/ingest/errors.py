"""Failure classification shared by every ingest flow.

The distinction that matters to a queue: some failures are transient (a
timeout, a 5xx, a dropped connection) and a retry may well succeed; others
are deterministic properties of the source (a 404, a deleted row, bytes that
are not a PDF) and retrying only burns a worker slot for the length of the
backoff ladder before failing anyway.

With DISPATCH_MAX_INFLIGHT slots in total, that distinction is not cosmetic:
a handful of deterministic failures retried at 30s + 120s each will starve
real ingests for minutes.
"""
from __future__ import annotations


class SourceError(RuntimeError):
    """The source itself is bad — a 404, no such host, not the media type we
    were promised, or a manifest row that no longer exists.

    Deterministic: retrying cannot change the outcome, so
    retry_unless_source_error() declines to retry these and the flow fails the
    source with this message as its readable reason.
    """


def retry_unless_source_error(task, task_run, state) -> bool:
    """Prefect retry_condition_fn: retry network blips, never a bad source."""
    try:
        state.result()
    except SourceError:
        return False
    except Exception:
        return True
    return False
