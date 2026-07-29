"""Ingest worker entrypoint — serves BOTH Prefect flows from one process.

    python -m src.worker

serve() registers the "ms-ingest-video/ingest" and "ms-ingest-paper/ingest"
deployments in Prefect Cloud (idempotent) and long-polls for scheduled runs —
outbound HTTPS only, no ports. The `limit` is Runner-global: videos and papers
share the same WORKER_CONCURRENCY execution slots, which matches the
dispatcher's single DISPATCH_MAX_INFLIGHT accounting. Scale horizontally by
running more replicas of this process.

Sample seeding is NOT done here — it's a one-shot startup gate (seed.py /
src/seeding.py) that the whole stack waits on, so the app never serves a
half-indexed corpus. This worker only handles user-registered sources.

Embedding goes to the warm CLIP service when CLIP_SERVICE_URL is set
(docker-compose default); unset, each run loads the model in-process.
"""
import os
import time

from prefect import serve
from prefect.deployments.runner import EntrypointType

from .db import init_schema
from .ingest.paper_pipeline import ingest_paper
from .ingest.pipeline import ingest_video


def main():
    init_schema()  # make sure migrations ran before consuming runs
    from .rag import vector_store
    vector_store.ensure_collection()  # up front, not mid-first-ingest
    # Fair scheduler (WFQ): admits pending sources round-robin across users so
    # one bulk uploader can't starve everyone else (src/dispatcher.py).
    from . import dispatcher
    dispatcher.start_in_background()
    limit = int(os.getenv("WORKER_CONCURRENCY", "2"))
    # serve() talks to Prefect Cloud on startup; a transient outage (e.g. a 503)
    # used to crash the worker permanently and stop the machine. Self-heal:
    # retry forever so a blip pauses ingest instead of killing the worker.
    while True:
        try:
            print(f"[worker] serving 'ms-ingest-video/ingest' + "
                  f"'ms-ingest-paper/ingest' (shared concurrency {limit})")
            # MODULE_PATH entrypoints, deliberately. The default FILE_PATH makes
            # the flow-run subprocess exec the pipeline file as a standalone
            # script (prefect load_script_as_module) — where `from .. import db`
            # dies with "attempted relative import beyond top-level package".
            # A module entrypoint imports src.ingest.* as the real package, the
            # same way this worker itself does.
            serve(
                ingest_video.to_deployment(
                    name="ingest", entrypoint_type=EntrypointType.MODULE_PATH),
                ingest_paper.to_deployment(
                    name="ingest", entrypoint_type=EntrypointType.MODULE_PATH),
                limit=limit,
            )
            break  # clean shutdown
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"[worker] serve crashed: {type(exc).__name__}: {exc} — retrying in 15s")
            time.sleep(15)


if __name__ == "__main__":
    main()
