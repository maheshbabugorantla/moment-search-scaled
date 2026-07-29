"""Capture the baseline ingest behaviour from Prefect Cloud as markdown.

REC-300 asks for "the queue behaviour" — the reference every later change must
not break. A screenshot of the Prefect run view can't be diffed and can't be
regenerated, so this pulls the same facts from the API instead: flow runs, their
task runs, per-task durations, and the retry count on each.

Run it against the live stack (it reads PREFECT_API_URL/PREFECT_API_KEY from the
app config, same as scripts/check_env.py):

    docker compose run --rm --no-deps -e PYTHONPATH=/app \
        -v "$PWD/scripts:/app/scripts" api python /app/scripts/capture_baseline.py

Writes markdown to stdout; redirect it where you want it.
"""
from __future__ import annotations

import asyncio
import sys

from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import TaskRunFilter, TaskRunFilterFlowRunId

# The flow names the video pipeline creates (src/ingest/pipeline.py).
INGEST_PREFIX = "ingest-"


def _secs(td) -> str:
    return f"{td.total_seconds():.1f}s" if td else "—"


async def collect(limit: int = 50) -> list[dict]:
    async with get_client() as client:
        runs = [r for r in await client.read_flow_runs(limit=limit)
                if r.name.startswith(INGEST_PREFIX)]
        out = []
        for r in runs:
            tasks = await client.read_task_runs(
                task_run_filter=TaskRunFilter(
                    flow_run_id=TaskRunFilterFlowRunId(any_=[r.id])),
            )
            tasks.sort(key=lambda t: t.start_time or t.expected_start_time)
            out.append({"run": r, "tasks": tasks})
        return out


def render(rows: list[dict]) -> str:
    lines: list[str] = []
    add = lines.append

    add("## Observed flow runs\n")
    add("| Flow run | State | Wall clock | Tasks |")
    add("| --- | --- | --- | --- |")
    for row in rows:
        r = row["run"]
        add(f"| `{r.name}` | {r.state_type.name} | {_secs(r.total_run_time)} "
            f"| {len(row['tasks'])} |")

    add("\n## Per-task detail\n")
    add("`run_count` is Prefect's attempt counter — 1 means the task succeeded "
        "first try, >1 means it retried.\n")
    for row in rows:
        r = row["run"]
        add(f"\n**`{r.name}`** — {r.state_type.name}, {_secs(r.total_run_time)} "
            f"(`{r.id}`)\n")
        add("| Task | State | Duration | run_count |")
        add("| --- | --- | --- | --- |")
        for t in row["tasks"]:
            add(f"| `{t.name}` | {t.state_type.name} | {_secs(t.total_run_time)} "
                f"| {t.run_count} |")
    return "\n".join(lines)


def main() -> int:
    rows = asyncio.run(collect())
    if not rows:
        print("No ingest flow runs found in Prefect Cloud.", file=sys.stderr)
        return 1
    print(render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
