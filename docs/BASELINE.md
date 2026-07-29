# Baseline — the video pipeline before `kind` exists

The reference behaviour every later change must not break. Captured on the
untouched app with four talks ingested at full quality, so that when paper and
deck ingestion land there is something concrete to diff against.

Evidence comes from the Prefect Cloud API rather than a screenshot, via
`scripts/capture_baseline.py` — it can be regenerated and diffed, a screenshot
can be neither.

## Corpus

Four LLM talks (`src/samples.py`), registered through `POST /api/videos` with
`SEED_SAMPLE_VIDEOS=false`, so every one went through the ordinary user path
rather than the seed gate.

| Video | Talk | Frames |
| --- | --- | --- |
| `yt_LPZh9BOjkQs` | 3Blue1Brown — LLMs explained briefly (8m) | 156 |
| `yt_wjZofJX0v4M` | 3Blue1Brown — Transformers, the tech behind LLMs (27m) | 307 |
| `yt_eMlx5fFNoYc` | 3Blue1Brown — Attention in transformers (26m) | 292 |
| `yt_zjkBMFhNj_g` | Karpathy — [1hr Talk] Intro to LLMs (60m) | 72 |

Frame count tracks scene changes, not duration — which is why the 60-minute
Karpathy talk (a static slide deck, filmed) yields 72 frames while a 27-minute
3Blue1Brown animation yields 307. Worth remembering when deck ingestion starts
competing for the same index.

**Index totals:** 827 image vectors (`moments`, CLIP 512-dim) + 354 text vectors
(`moments_text`, bge 384-dim) = 1181.

## The status lifecycle

The sequence papers and decks have to copy. Every write site is in
`src/ingest/pipeline.py` unless noted.

```
pending  →  queued  →  fetching  →  sampling  →  embedding  →  indexed
```

| Status | Written by | Meaning |
| --- | --- | --- |
| `pending` | `db.upsert_pending` on register (`api/videos.py`) | in the waiting line |
| `queued` | `db.wfq_claim` (`db.py:220`) | the dispatcher admitted it |
| `fetching` | `t_fetch` (`pipeline.py:45`) | acquiring the source |
| `sampling` | `t_sample` (`pipeline.py:70`) | extracting frames; `progress` 0→1 |
| `embedding` | `t_embed_index` (`pipeline.py:98`) | CLIP + upsert; `progress` 0→1 |
| `indexed` | `t_embed_index` (`pipeline.py:118`) | **visual branch complete** — see below |

Two terminal alternates:

* `skipped` — `t_fetch` found an existing `indexed` row with the same
  `source_hash` (`pipeline.py:62`). Content-level idempotency.
* `failed` — any unhandled exception in the flow body (`pipeline.py:173`).

The waiting line lives in Postgres, not in the queue. `db.wfq_claim` is an
atomic `UPDATE ... WHERE status='pending' RETURNING`, so several dispatchers can
race and each row is still handed out once.

### `indexed` does not mean both branches are indexed

`t_embed_index` writes `indexed` at `pipeline.py:118`. The transcript task runs
*after* it, and swallows every exception into "visual-only"
(`pipeline.py:152-154`). So:

* a video reports `indexed` when only the **visual** branch has completed;
* a transcript failure — a YouTube bot-check, a video with no captions, an
  upload with no caption track — is invisible in the status;
* search silently degrades to visual-only, which looks like worse ranking rather
  than a broken ingest.

This is not hypothetical: re-registering the four talks once took `moments_text`
from 354 to 0 while all four rows reported `indexed`. `YT_COOKIES_FILE` is
unset, so the trigger is still live. It is the reason `tests/conftest.py`
forbids re-registering a corpus video, and it is worth fixing when the manifest
gains per-stage state (REC-310, REC-318).

## Observed flow runs

Four tasks per run, no retries anywhere — `run_count=1` throughout. This is the
clean-path baseline; the retry behaviour it *would* show under failure is
`t_fetch` 2, `t_embed_index` 2, `t_transcript` 1, and `t_sample` **none**.

| Flow run | State | Wall clock | fetch | sample | embed-index | transcript |
| --- | --- | --- | --- | --- | --- | --- |
| `ingest-yt_LPZh9BOjkQs` | COMPLETED | 19.4s | 8.0s | 3.0s | 4.9s | 2.0s |
| `ingest-yt_wjZofJX0v4M` | COMPLETED | 27.0s | 10.0s | 5.6s | 7.9s | 2.0s |
| `ingest-yt_eMlx5fFNoYc` | COMPLETED | 26.2s | 9.7s | 5.4s | 7.5s | 1.9s |
| `ingest-yt_zjkBMFhNj_g` | COMPLETED | 23.0s | 10.7s | 5.8s | 3.1s | 2.0s |

Flow run IDs, for the Prefect Cloud UI:

* `06a68cb5-e812-7ef8-8000-62455041cb95` — `ingest-yt_LPZh9BOjkQs`
* `06a68cb6-34d7-7038-8000-abf7d6e24840` — `ingest-yt_wjZofJX0v4M`
* `06a68cb7-8827-7565-8000-9e392208d3dc` — `ingest-yt_eMlx5fFNoYc`
* `06a68cb8-a4cf-7329-8000-7ba0853f7730` — `ingest-yt_zjkBMFhNj_g`

Shape of the numbers, since the SLA gate will be argued against them: fetch
dominates and is network-bound; embedding scales with frame count (3.1s for 72
frames, 7.9s for 307); transcript is ~2s flat because it is a subtitle download
plus a small bge batch. Ingest is roughly 20–27s per talk end to end, and the
four were dispatched about 10–20s apart rather than all at once — the WFQ
dispatcher admitting in fair order, not a thundering herd.

## The read path

`POST /api/ask` returns a single JSON body. There is no SSE stream and no
`/ask_stream` route, whatever the assignment brief says.

Citation payload, verbatim from a live query:

```
n, video_id, title, url, source, ms, timestamp, idx,
thumbnail, media_url, deeplink, score, transcript, modalities
```

No speaker or diarization field — the text branch is yt-dlp subtitles, which
carry no speaker labels. `modalities` records which branches agreed on a moment
(`frame`, `text`, or both), and both-agree windows get `CROSS_MODAL_BOOST`.

`deeplink` seeks the player to `ms // 1000`. That is the product: the citation
is a *moment*, not a document. Pages and slides have to arrive as locators of
the same standing (REC-332, REC-314).

## Regenerating this

```bash
docker compose run --rm --no-deps -e PYTHONPATH=/app \
    -v "$PWD/scripts:/app/scripts" api python /app/scripts/capture_baseline.py
docker compose run --rm tests      # 18 contract assertions, all green
```
