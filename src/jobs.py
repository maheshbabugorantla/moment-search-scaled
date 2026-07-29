"""Prefect Cloud trigger layer — the API schedules runs, workers execute them.

Three flows ("ms-ingest-video", "ms-ingest-paper" and "ms-ingest-post" — the
"ms-" prefix keeps them distinct from the digital-twin-akash flow living in the
same Prefect workspace), each with an "ingest" deployment registered by
worker.py's serve(). The API never imports the pipelines or their heavy deps
(torch, ffmpeg, pymupdf) — it just asks Prefect Cloud to schedule a run; any
live worker picks it up. Retries/backoff live on the flows' tasks
(src/ingest/pipeline.py, paper_pipeline.py, post_pipeline.py); failed runs are
visible + retryable in the Prefect Cloud UI.
"""
from __future__ import annotations

from prefect.deployments import run_deployment

INGEST_DEPLOYMENT = "ms-ingest-video/ingest"
PAPER_DEPLOYMENT = "ms-ingest-paper/ingest"
POST_DEPLOYMENT = "ms-ingest-post/ingest"


def enqueue_video(video_id: str, user_id: str) -> str:
    """Schedule the ingest flow for one video. Returns the Prefect flow-run id."""
    flow_run = run_deployment(
        name=INGEST_DEPLOYMENT,
        parameters={"video_id": video_id, "user_id": user_id},
        timeout=0,  # fire-and-forget: don't block the API waiting for the run
        flow_run_name=f"ingest-{video_id}",
    )
    return str(flow_run.id)


def enqueue_paper(doc_id: str, user_id: str) -> str:
    """Schedule the paper ingest flow for one document."""
    flow_run = run_deployment(
        name=PAPER_DEPLOYMENT,
        parameters={"doc_id": doc_id, "user_id": user_id},
        timeout=0,
        flow_run_name=f"ingest-paper-{doc_id}",
    )
    return str(flow_run.id)


def enqueue_post(doc_id: str, user_id: str) -> str:
    """Schedule the markdown-post ingest flow for one document."""
    flow_run = run_deployment(
        name=POST_DEPLOYMENT,
        parameters={"doc_id": doc_id, "user_id": user_id},
        timeout=0,
        flow_run_name=f"ingest-post-{doc_id}",
    )
    return str(flow_run.id)
