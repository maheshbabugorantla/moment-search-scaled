# Architecture notes — what the base app actually does

Read-through of `traversaal-ai/momentsearch` at commit `8526743`, written for
Assignment 3 (multi-source ingestion). **Describes what exists**, not what we
intend to build. Every claim cites a file and line.

> **Read this before touching anything.** The assignment README describes a
> slightly different system — different port, different route names, a semantic
> chunker that does not exist, and "one shared index" where there are two
> collections. Where they disagree, *this* document matches the code.

---

## 1. Processes and topology

`docker-compose.yml` runs four services off one image:

| Service | Command | Role |
|---|---|---|
| `clip` | `uvicorn src.clip_service:app --port 8001` | one warm CLIP model behind HTTP; api+worker send batches here instead of each loading torch |
| `seed` | `python -m src.seed` | **one-shot startup gate** — indexes 4 sample talks, then exits |
| `api` | default CMD | FastAPI on **:8000** |
| `worker` | `python -m src.worker` | serves the Prefect deployment; runs the ingest flow |

`api` and `worker` both `depend_on: seed: service_completed_successfully`. **If
the seed fails, neither ever starts** — see §9, risk 1.

`src/app.py:44-46` mounts exactly two routers; `src/app.py:29-41` is the lifespan
hook that runs `db.init_schema()` and creates both Qdrant collections at boot,
tolerating a Qdrant outage rather than failing to start.

---

## 2. The HTTP surface as it exists today

**Port 8000** (`src/app.py:15`), not 8100.

| Method | Path | Auth | Source |
|---|---|---|---|
| POST | `/api/videos` | Bearer | `api/videos.py:117` — register, returns **202** |
| POST | `/api/videos/presign` | Bearer | `api/videos.py:66` |
| PUT | `/api/videos/{id}/content` | Bearer | `api/videos.py:86` — local-dev upload |
| GET | `/api/videos` | — | `api/videos.py:166` — list, tenant-scoped |
| GET | `/api/videos/{id}` | — | `api/videos.py:171` |
| POST | `/api/videos/{id}/retry` | Bearer | `api/videos.py:179` |
| DELETE | `/api/videos/{id}` | Bearer | `api/videos.py:191` |
| POST | `/api/ask` | — | `api/search.py:139` — **JSON, not SSE** |
| GET | `/api/health`, `/api/config` | — | `api/search.py:37,42` |
| GET/PUT/DELETE | `/api/llm`, `/api/llm/test` | Bearer (writes) | `api/search.py:92-127` |
| GET | `/`, `/get-started` | — | `api/search.py:227,232` — the UI |

There is **no** `/admin/*` namespace and **no** `/ask_stream`. The router prefix
is fixed at `api/videos.py:36`.

**Auth** is `require_auth` (`api/videos.py:44-48`): compares the `Authorization`
header to `f"Bearer {ADMIN_TOKEN}"`, raising `HTTPException(401, ...)`. Note
`api/videos.py:45` — **an empty `ADMIN_TOKEN` disables auth entirely**.

**Errors** are bare `HTTPException(status, "message")` throughout, so responses
are FastAPI's default `{"detail": "..."}`. Codes in use: `400` (`videos.py:54,
122, 142`), `403` (`videos.py:93, 130, 197`), `404` (`videos.py:133, 175`), `413`
(`videos.py:69, 103, 136`), `415` (`videos.py:71`), `502` (`search.py:120`, LLM
call failure — the only 502 in the codebase). There is no shared error handler
and no error envelope beyond `detail`.

**Tenancy:** every request resolves a tenant from the `X-User-Id` header,
defaulting to `"default"` (`api/videos.py:51-55`, `config.py:47`). This is not
cosmetic — see §6.

---

## 3. Write path: register → dispatch → queue → worker

```
POST /api/videos                      api/videos.py:117
  └─ db.upsert_pending(...)           db.py:78    row status='pending'
  └─ if ENABLE_FAIR_DISPATCH (default true):
       return {"video_id", "status":"pending"}    api/videos.py:147
     else:
       jobs.enqueue_video(...)        api/videos.py:148

dispatcher thread (in the WORKER process, worker.py:31)
  └─ every DISPATCH_INTERVAL_S (3s)   dispatcher.py:49-57
       slots = DISPATCH_MAX_INFLIGHT - db.count_inflight()   dispatcher.py:33
       db.wfq_claim(slots)            db.py:190   'pending' -> 'queued', atomic
       jobs.enqueue_video(id, user)   dispatcher.py:39

jobs.enqueue_video                    jobs.py:18-26
  └─ run_deployment("ms-ingest-video/ingest", parameters={video_id, user_id},
                    timeout=0)        jobs.py:20-25   fire-and-forget

worker.py:39  ingest_video.serve(name="ingest", limit=WORKER_CONCURRENCY)
  └─ long-polls Prefect Cloud, executes runs in subprocesses
```

Two things worth internalizing:

1. **The API never enqueues by default.** `ENABLE_FAIR_DISPATCH` defaults `True`
   (`config.py:122`), so registration only writes a `pending` row; a
   weighted-fair-queueing dispatcher inside the *worker* admits it later. The
   admission decision is `db.wfq_claim` (`db.py:190-225`), which ranks each
   user's pending rows by age and takes everyone's oldest first, so one bulk
   uploader cannot starve others. The claim is atomic
   (`UPDATE ... WHERE status='pending' RETURNING`, `db.py:218-225`).
2. **Concurrency is capped by us, not by Prefect.**
   `DISPATCH_MAX_INFLIGHT` defaults to `WORKER_CONCURRENCY` = **2**
   (`config.py:126`). Each worker replica starts its own dispatcher
   (`worker.py:31`), and `dispatcher.py:18-20` documents that N dispatchers may
   mildly over-admit. Any throughput measurement is bounded by this knob.

Registering the *same* video id again resets the row to `pending`
(`db.py:86-91`), which is how `/retry` works (`api/videos.py:184`).

---

## 4. The ingest flow

`src/ingest/pipeline.py` — one Prefect flow, four tasks.

| Task | Line | Retries | What it does |
|---|---|---|---|
| `t_fetch` | 38 | `retries=2`, delay `[30,120]` | yt-dlp or bucket download → scratch file; hashes it; duplicate check |
| `t_sample` | 67 | **none** | ffmpeg keyframes → pHash dedup → thumbnails to storage |
| `t_embed_index` | 95 | `retries=2`, delay `60` | CLIP-embed frames in `CLIP_BATCH` batches → Qdrant upsert |
| `t_transcript` | 123 | `retries=1`, delay `30` | YouTube captions → time chunks → bge → text collection |

Flow: `ingest_video(video_id, user_id)` at `pipeline.py:157-177`, named
`"ms-ingest-video"`, `timeout_seconds=3600`. **The flow itself has no
`retries=`** — only the tasks do.

**Which stages are generic over "chunks" (reusable for papers/decks):**

- `t_transcript` (`pipeline.py:123-154`) is the model to copy. It is the only
  path that turns *text* into vectors: `embed_docs(...)` →
  `vector_store.upsert_chunks(...)` with a payload carrying `text`, `t_start`,
  `t_end`, `ms`, `modality:"text"` (`pipeline.py:144-149`).
- `embed_docs` / `embed_query` (`rag/embeddings.py:175,188`) are fully generic —
  they take `list[str]` and dispatch to OpenAI, the remote CLIP service, or
  in-process bge based on config. **Reusable unchanged.**
- `vector_store.upsert_chunks` (`vector_store.py:177-187`) is generic over
  payload dicts. **Reusable**, though its point-ID scheme hard-codes `video_id`
  (see §5).
- `t_sample` and `t_embed_index` are frame/video-specific (ffmpeg, JPEG bytes,
  CLIP image encoder). Not reusable for documents.

**There is no LLM "enrichment" stage.** The assignment's
`chunk → enrich → embed` describes a pipeline this repo does not have.

**There is no semantic chunker.** `src/rag/chunk.py` does not exist. The only
chunking is `chunk_cues` (`ingest/transcript.py:82-101`), which groups caption
cues into ~`TRANSCRIPT_CHUNK_SECONDS` (20s) windows by *timestamp arithmetic*.
It takes `[{text, t_start, t_end}]` and cannot chunk a PDF.

### Status lifecycle (real)

`config.py:110-113`:

```
pending → queued → fetching → sampling → embedding → indexed | skipped | failed
INFLIGHT_STATUSES = (queued, fetching, sampling, embedding)
```

Not the assignment's `parsing → chunking → enriching → embedding`. `skipped`
means duplicate content for that tenant.

### Every place status is written

| Site | Value |
|---|---|
| `db.upsert_pending` (`db.py:85,91`) | `pending` (insert and on-conflict reset) |
| `db.wfq_claim` (`db.py:220`) | `queued` |
| `t_fetch` (`pipeline.py:45,53,57`) | `fetching` |
| `t_fetch` (`pipeline.py:62`) | `skipped` |
| `t_sample` (`pipeline.py:70`) | `sampling` |
| `t_embed_index` (`pipeline.py:98`) | `embedding` |
| `t_embed_index` (`pipeline.py:118`) | **`indexed`** |
| `ingest_video` except (`pipeline.py:173`) | `failed` |
| `dispatcher.dispatch_once` (`dispatcher.py:42`) | back to `pending` if Prefect unreachable |
| `api/videos.py:184` (`/retry`) | `pending` |

Progress is a separate `REAL` column 0..1 via `db.set_progress`
(`db.py:120-123`), written every 25 thumbnails (`pipeline.py:86-87`) and per
embed batch (`pipeline.py:117`). **It is a float 0..1, not an integer percent.**

### Crash-safety, as already implemented

`t_embed_index` sets `indexed` at `pipeline.py:118` — *after* the upsert loop
(`pipeline.py:103-117`) has completed. Point IDs are `uuid5` of
`f"{video_id}:{frame_idx}"` (`vector_store.py:76-77`), and the task deletes the
video's existing points before re-upserting (`pipeline.py:100`). **So
ack-after-upsert and idempotent replay already hold for video.** For Assignment
3 this is a property to *preserve*, not to add.

---

## 5. Qdrant — two collections, not one

`src/rag/vector_store.py`. There are **two** collections, created by
`ensure_collection` (`:129`) and `ensure_text_collection` (`:134`):

| Collection | Default name | Vectors | Dim | Written by |
|---|---|---|---|---|
| visual | `moments` (`config.py:266`) | CLIP image | 512 for `clip-ViT-B-32` (`vector_store.py:47-52`) | `upsert_frames` (`:139`) |
| text | `moments_text` (`config.py:171`) | bge or OpenAI | 384 / 1536 (`config.py:189`) | `upsert_chunks` (`:177`) |

They cannot be merged casually: **different embedding models with different
dimensions**. A Qdrant collection has one vector size. The system reconciles them
at query time by rank fusion (§6), not in the index. "Papers, decks, and videos
land in the same collection" therefore means: *the same collection as the
transcript branch*, `moments_text`, embedded with the same text model.

**Point IDs**
- frames: `uuid5(NAMESPACE_URL, f"{video_id}:{frame_idx}")` (`:76-77`)
- text chunks: `uuid5(NAMESPACE_URL, f"{video_id}:text:{i}")` (`:182`)

Both deterministic — re-runs overwrite. Note both are keyed on `video_id`; a
document id would slot in unchanged, but the `:text:` namespace is shared, so a
document's chunks must not collide with a video's.

**Payload shape**
- frame (`pipeline.py:110-114`): `user_id, video_id, ms, idx, modality:"frame",
  t_start, t_end, embed_version`
- text (`pipeline.py:146-149`): `user_id, video_id, modality:"text", t_start,
  t_end, ms, text, embed_version`

Titles and URLs are **not** in Qdrant — they are joined from Postgres at answer
time via `db.videos_by_ids` (`db.py:164-170`).

**Filtering:** `_user_filter` (`vector_store.py:80-88`) puts `user_id` in a
`must` clause on **every** search, upsert-delete and delete. Optionally adds
`video_id` / `video_ids` for UI scoping. So: *yes, retrieval always filters* —
on tenant. `user_id` has a tenant payload index (`:112-115`), `video_id` a
keyword index (`:122-124`).

`delete_video` (`:212-219`) purges from **both** collections.

Search failures on a missing collection are swallowed into `[]`
(`:169-171`, `:206-208`) so an empty deployment answers "no moments" instead of
500ing.

---

## 6. Read path: `/api/ask`

`api/search.py:139` → `rag_search.ask` (`rag/search.py:211-246`).

```
ask()
 └─ retrieve()                         rag/search.py:103
      ├─ visual: embed_text(q) → vector_store.search(BRANCH_TOP_K=20)      :115
      ├─ text:   embed_query(q) → vector_store.search_text(BRANCH_TOP_K)   :123
      ├─ _fuse(vhits, thits)[:TOP_K]                                       :128
      └─ build citation dicts                                             :139-154
 └─ if no citations → "no relevant moments", abstained                    :218
 └─ Gate 1: if best_visual < 0.2 AND best_text < 0.35 → ABSTAIN,
    without calling the LLM                                               :225-229
 └─ if no LLM configured → _fallback_answer (retrieval-only summary)      :233
 └─ _build_moments → llm.answer → _validate_citations                     :240-242
```

**Fusion** — `_fuse` (`rag/search.py:31-74`) is the critical function:

- each branch is ranked independently, scored `rrf = 1/(RRF_K + rank)` (`:44`)
- hits are bucketed into "windows": same `video_id` **and** within
  `FUSION_WINDOW_S` (15s) of each other (`:52-53`)
- **only the best hit per modality is kept per window** (`:64-65`), deliberately,
  to stop a burst of near-identical frames outscoring a genuine match
- a window with both modalities gets `CROSS_MODAL_BOOST` ×1.5 (`:71-72`)

The window's time comes from `t = float(h.get("t_start", h.get("ms", 0)/1000.0))`
(`:43`). **A chunk with neither field lands at `t = 0.0`.** See §9, risk 2.

**Citation object** (`rag/search.py:139-154`) — the current shape:

```python
{"n", "video_id", "title", "url", "source", "ms", "timestamp", "idx",
 "thumbnail", "media_url", "deeplink", "score", "transcript", "modalities"}
```

There is no `kind` and no `locator`. The temporal locator is flat: `ms` +
`timestamp` + `deeplink` (`_deeplink`, `:77-82`, appends `?t=<secs>` to a
YouTube URL or `#t=` to the local media route).

**Grounding already exists, partially.** `_validate_citations`
(`rag/search.py:170-179`) regex-strips any `[n]` reference the model invented
beyond the number of citations supplied. Combined with Gate 1 and `ABSTAIN`
(`:22`), the "never fabricate" machinery is present for *citation indices* —
just not for locator fields, which the LLM never sees or emits today.

---

## 7. Storage and config

`src/storage.py` — one S3 client for aws/gcp/flyio, Google SDK for
`gcp_native`, plain filesystem under `./data` for `local` (`config.py:53`). Key
layout (`config.py:93-97`):

```
uploads/{user_id}/{video_id}.{ext}       raw source
frames/{user_id}/{video_id}/NNNNNN.jpg   thumbnails
```

Helpers: `put_bytes`, `get_bytes`, `download_to`, `upload_file`, `presign_put`,
`presign_get`, `head`, `list_keys`, `delete_prefix`, `delete_key`,
`local_path`, `presign_capable`.

Knobs that will matter later: `TOP_K=6` (`config.py:275`), `BRANCH_TOP_K=20`
(`:210`), `RRF_K=60` (`:203`), `FUSION_WINDOW_S=15` (`:205`),
`CROSS_MODAL_BOOST=1.5` (`:208`), `CONFIDENCE_THRESHOLD=0.2` /
`TEXT_CONFIDENCE_THRESHOLD=0.35` (`:281-282`), `MAX_FRAMES=400` (`:135`),
`DISPATCH_MAX_INFLIGHT=2` (`:126`).

---

## 8. Where `kind` would have to branch

Purely as a map of seams — not a design:

| Seam | File:line | Why |
|---|---|---|
| register endpoint | `api/videos.py:117` | only accepts a YouTube URL or a presigned upload key; `_YT_RE` gate at `:120-122` |
| manifest schema | `db.py:37-57` | `ms_videos` has no `kind`; `source` is `youtube\|upload` |
| deployment name | `jobs.py:15` | one constant, one deployment |
| worker serve | `worker.py:39` | serves exactly one flow |
| flow entry | `pipeline.py:157` | `ingest_video(video_id, user_id)` — stage list is fixed |
| chunk producer | *(absent)* | nothing parses a document |
| upsert | `vector_store.py:177-187` | generic, but IDs keyed `{video_id}:text:{i}` |
| fusion | `rag/search.py:43,52-53` | windows keyed on `video_id` + time |
| citation build | `rag/search.py:139-154` | flat `ms`/`timestamp`, no `kind`/`locator` |
| UI render | `ui/index.html` | renders timestamp + thumbnail per citation |

---

## 9. Risks found while reading

**1. The seed gate can block the entire stack.**
`seeding.seed_to_completion` (`seeding.py:50-96`) tries 3 passes over the 4
sample YouTube videos and returns `False` if any remain unindexed
(`:90-94`); `seed.py:17` turns that into exit 1; compose then never starts
`api` or `worker`. `SEED_SAMPLE_VIDEOS=false` short-circuits to `True`
(`seeding.py:53-55`). *Verified working on this machine — a metadata probe of
the first sample returned 23 formats with no cookies.*

**2. `_fuse` collapses a non-temporal source into one citation.**
Because `t` defaults to `0.0` when a payload has no `t_start`/`ms`
(`rag/search.py:43`), every chunk of one document falls inside the same 15s
window for the same source id (`:52-53`), and only the best hit per modality
survives (`:64-65`). Result: **one citation per document, regardless of how many
pages matched.** Page- and slide-level granularity is destroyed at fusion time,
*after* retrieval got it right. Anything that needs "p.4 *and* slide 12" from one
query has to address this function.

**3. A crashed run leaves a row stuck in-flight forever.**
The flow has no `retries=` (`pipeline.py:157`), and `db.wfq_claim` only ever
selects `WHERE status = 'pending'` (`db.py:209`). A worker killed during
`sampling` leaves the row at `sampling`, which is in `INFLIGHT_STATUSES`
(`config.py:113`) — so `count_inflight()` (`db.py:180-187`) counts it against
`DISPATCH_MAX_INFLIGHT` permanently, shrinking capacity, and nothing re-admits
it. There is no reaper. *Needs empirical confirmation of what Prefect does with
the run, but no code path re-queues a non-`pending` row.*

**4. `t_sample` has no retry policy** (`pipeline.py:67`) while its neighbours do
— an ffmpeg hiccup fails the whole run.

**5. Multi-tenancy is invisible to the graders.** Every point carries `user_id`
and every query filters on it (§5), sourced from `X-User-Id` defaulting to
`"default"` (`config.py:47`). `bench.py` and `eval.py` send no such header, so
they operate as tenant `default` — fine, as long as every ingest path resolves
the same tenant. Anything that writes with a different tenant becomes invisible
to search.

**6. Empty `ADMIN_TOKEN` silently disables auth** (`api/videos.py:45`) — a `401`
test passes vacuously if the token is unset.

**7. Registration returns `video_id`, not `id`** (`api/videos.py:147`), and the
fair-dispatch path returns no `flow_run_id`.

**8. A re-ingest can silently destroy the text branch.** *Observed, not
theorised.* `t_embed_index` calls `vector_store.delete_video`
(`pipeline.py:100`), which purges the video from **both** collections
(`vector_store.py:212-219`). `t_transcript` then re-indexes the text — but it is
best-effort by design and swallows every failure into "visual-only"
(`pipeline.py:152-154`). Re-registering the four sample talks triggered YouTube's
bot-check on the *subtitle* fetch (the video downloads themselves succeeded), so
all four transcript stages returned 0 and the run still finished `indexed`. Net
effect: `moments_text` went from 354 points to **0**, every source reported
healthy, and search silently degraded to visual-only. Recovered by re-running
just the transcript stage once the rate-limit cleared.

Two consequences for this assignment:

- `moments_text` is where paper and deck chunks will live. A stage that can empty
  it while reporting success is a recall failure with no error to find. Anything
  that re-ingests a source needs the delete to be scoped, or the re-index to be
  mandatory rather than best-effort.
- `indexed` currently means "the visual branch upserted", not "everything this
  source should contribute is in the index". Any definition of done for documents
  has to be stricter than the one video uses.

The documented mitigation for the bot-check is cookies —
`YT_COOKIES_FILE=/app/data/cookies.txt` (`config.py:229`, `.env.example:119-127`);
worth setting before any large YouTube backfill.

---

## 10. Consequences for the Assignment-3 plan

Corrections these findings force on the tracker (Linear project *FDE A3*):

- **REC-302 (1.1, contract test)** asserts `POST /admin/videos` → `{id, status}`.
  Real: `POST /api/videos` → `{"video_id", "status"}` (`api/videos.py:117,147`).
  Both path and key are wrong; the test would fail on the untouched baseline.
- **REC-308 (2.2, chunking)** says reuse `src/rag/chunk.py`. That file does not
  exist and no semantic chunker exists (§4).
- **REC-309 (2.3)** treats ack-after-upsert as new work. It already holds
  (`pipeline.py:118`); the criterion is *preserve*.
- **REC-315 (4.2, grounding)** — `_validate_citations` (`rag/search.py:173`) and
  the two-threshold gate already cover invented citation *indices*; the gap is
  locator fields.
- **REC-318 (5.1)** scopes resilience to checkpointing. Risk 3 says a
  requeue/reaper path is needed before checkpointing matters.
- **Epic 4** has no issue covering `_fuse` (risk 2), which is the hard blocker
  for multi-locator answers.
- **Epic 1** must decide `/admin/*` versus the existing `/api/*`: `bench.py`
  defaults to `:8100` and `eval/rubric.json` names `/admin/documents`,
  `/admin/sources`, `/ask_stream` — none of which exist (§2).
- **`/ask_stream` does not exist at all.** `/api/ask` is a plain JSON POST
  (`api/search.py:139`); the SSE trace-then-citations-then-answer stream the
  assignment describes would be new work.
