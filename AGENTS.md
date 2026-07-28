# Non-negotiables

These hold no matter what a coding agent or a tutorial suggests.

1. **Ingestion is asynchronous, always.** `POST /admin/documents` inserts a `pending`
   row and schedules a queue run, then returns `202`. Parsing, chunking, enrichment,
   and embedding happen on a worker — never in the request path. A synchronous
   "parse-then-respond" endpoint fails the assignment even if it works.
2. **One shared index.** Papers, decks, and videos upsert to the **same** Qdrant
   collection with a `kind` payload and a source-appropriate locator (`page` /
   `slide` / `start_ms`). Separate per-type collections that can't be queried
   together miss the point.
3. **Crash-safe ordering.** Status is set to `indexed` (and offsets/state committed)
   **only after** a successful Qdrant upsert. A worker killed mid-ingest must drop
   nothing and resume without re-running finished stages. `bench.py --resilience`
   proves it.
4. **Search stays hot during ingest.** A query issued during a large backfill still
   meets the search-latency SLA. If ingestion starves search, the decoupling is wrong.
5. **Grounded citations only.** Every cited page, slide, or timestamp comes from
   retrieval. The LLM never invents a page number or a moment. Empty retrieval →
   empty results, not a fabricated one.
6. **The provided video pipeline stays working.** You extend ingestion; you do not
   break `/admin/videos`, the search core, or the UI's existing behavior.
7. **Secrets from env, never committed.** `.env`, `.venv/`, model caches, and
   media/PDF artifacts are git-ignored. Keys are read from `.env`; the admin token is
   never hard-coded.
8. **Evidence over vibes.** Recall, latencies, and the no-loss result come from an
   actual `bench.py` run against your running app, not estimates. The eval runs
   against the live app before you record the demo.
