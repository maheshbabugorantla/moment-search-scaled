"""Central env-driven config — every knob in one place.

Same conventions as the digital-twin-akash service: module-level constants,
provider-neutral STORAGE_* credentials with AWS_* fallbacks, Prefect Cloud
read straight from PREFECT_API_URL / PREFECT_API_KEY by the SDK.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"  # local-provider storage root (dev only)


def _envbool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# --- Database (Neon Postgres) — videos manifest, source of truth ------------
DATABASE_URL = os.getenv("DATABASE_URL", "")

# --- API auth ----------------------------------------------------------------
# Bearer token required on every mutating endpoint (presign, register, delete,
# retry). The tenant is the X-User-Id header (default "default") — swap this
# for real per-user auth (JWT/Clerk) later without touching the data model:
# every bucket key, Postgres row, and Qdrant point is already user_id-tagged.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default")

# --- Object storage (videos + frame thumbnails) ------------------------------
# STORAGE_PROVIDER: local | aws | gcp | gcp_native | flyio
# aws/gcp/flyio share one boto3 S3 client (different endpoints); gcp_native
# uses Google's SDK + service-account JSON; local writes under ./data (dev).
STORAGE_PROVIDER = os.getenv("STORAGE_PROVIDER", "local").strip().lower()
STORAGE_BUCKET = (os.getenv("STORAGE_BUCKET", "")
                  or os.getenv("BUCKET_NAME", "")            # injected by `fly storage create`
                  or os.getenv("GCS_BUCKET_NAME", "")        # gcp_native conventions
                  or os.getenv("GOOGLE_CLOUD_BUCKET_NAME", ""))
STORAGE_ACCESS_KEY_ID = os.getenv("STORAGE_ACCESS_KEY_ID", "").strip() or os.getenv("AWS_ACCESS_KEY_ID", "").strip()
STORAGE_SECRET_ACCESS_KEY = os.getenv("STORAGE_SECRET_ACCESS_KEY", "").strip() or os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
AWS_REGION = os.getenv("STORAGE_REGION", "").strip() or os.getenv("AWS_REGION", "auto")
_PROVIDER_ENDPOINTS = {
    "aws": None,  # boto3 default
    "flyio": "https://fly.storage.tigris.dev",
    "gcp": "https://storage.googleapis.com",
}
STORAGE_ENDPOINT = os.getenv("AWS_ENDPOINT_URL_S3", "").strip() or _PROVIDER_ENDPOINTS.get(STORAGE_PROVIDER)


def gcs_service_account_info() -> dict:
    """Service-account JSON for STORAGE_PROVIDER=gcp_native, rebuilt from the
    GOOGLE_CLOUD_* env vars (the standard exploded-JSON convention)."""
    key = os.getenv("GOOGLE_CLOUD_PRIVATE_KEY", "").strip()
    # dotenv strips surrounding quotes locally, but `fly secrets import` keeps
    # them literally — strip defensively so the PEM is valid in both places.
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        key = key[1:-1]
    return {
        "type": "service_account",
        "project_id": os.getenv("GOOGLE_CLOUD_PROJECT_ID", ""),
        "private_key_id": os.getenv("GOOGLE_CLOUD_PRIVATE_KEY_ID", ""),
        "private_key": key.replace("\\n", "\n"),  # dotenv keeps \n literal inside quotes
        "client_email": os.getenv("GOOGLE_CLOUD_CLIENT_EMAIL", ""),
        "client_id": os.getenv("GOOGLE_CLOUD_CLIENT_ID", ""),
        "auth_uri": os.getenv("GOOGLE_CLOUD_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
        "token_uri": os.getenv("GOOGLE_CLOUD_TOKEN_URI", "https://oauth2.googleapis.com/token"),
        "auth_provider_x509_cert_url": os.getenv(
            "GOOGLE_CLOUD_AUTH_PROVIDER_X509_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
        "client_x509_cert_url": os.getenv("GOOGLE_CLOUD_CLIENT_X509_CERT_URL", ""),
        "universe_domain": os.getenv("GOOGLE_CLOUD_UNIVERSE_DOMAIN", "googleapis.com"),
    }


# Bucket key layout — every key is user-scoped (tenant isolation at the path level):
#   uploads/{user_id}/{video_id}.{ext}      raw uploaded video (presigned PUT target)
#   frames/{user_id}/{video_id}/NNNNNN.jpg  downscaled frame thumbnails (citations)
#   papers/{user_id}/{sha256}.pdf           fetched PDFs, content-addressed —
#                                           the same paper registered twice
#                                           resolves to one object
#   posts/{user_id}/{sha256}.md             fetched markdown, content-addressed
UPLOAD_KEY_PREFIX = "uploads/"
FRAME_KEY_PREFIX = "frames/"
PAPER_KEY_PREFIX = "papers/"
POST_KEY_PREFIX = "posts/"

# The pseudo-anchor for content before a post's first heading. Shared
# vocabulary rather than a parser detail: the read path has to recognise it to
# know a citation cannot deeplink to a fragment (nothing renders `#_top`).
TOP_ANCHOR = "_top"

# --- Paper ingest --------------------------------------------------------------
# Cheap guard on the fetch stage: a "PDF" bigger than this fails the source with
# a readable reason instead of filling the worker's disk.
MAX_PAPER_MB = _int("MAX_PAPER_MB", 100)
# Page-aware chunking (rag/chunk.py) — characters per chunk and the overlap
# carried across a forced split inside one long paragraph.
PAPER_CHUNK_CHARS = _int("PAPER_CHUNK_CHARS", 1400)
PAPER_CHUNK_OVERLAP = _int("PAPER_CHUNK_OVERLAP", 200)
# Chunks per bge embed call in the paper flow (transcript chunks go in one call
# because a talk has few; a 60-page paper does not).
PAPER_EMBED_BATCH = _int("PAPER_EMBED_BATCH", 64)

# --- Post ingest (markdown) -----------------------------------------------------
# Much smaller cap than a paper: a long-form essay is tens of KB of text, so
# anything near a megabyte is a mis-export (a whole archive, or a saved page
# with its assets inlined) and failing fast beats indexing it.
MAX_POST_MB = _int("MAX_POST_MB", 5)
# Section-bounded chunking (rag/chunk.py chunk_markdown). Same defaults as the
# paper knobs — the packer is shared; only the locator differs.
POST_CHUNK_CHARS = _int("POST_CHUNK_CHARS", 1400)
POST_CHUNK_OVERLAP = _int("POST_CHUNK_OVERLAP", 200)
POST_EMBED_BATCH = _int("POST_EMBED_BATCH", 64)

# Image worthiness (src/ingest/post_images.py). Off flips posts to text-only
# without touching the flow — the classifier is the newest code here, so it
# gets a switch.
POST_INDEX_IMAGES = _envbool("POST_INDEX_IMAGES", True)
# Heuristic layer. Only the unambiguous shapes are hard drops: an animated GIF
# is a reaction image, and anything thinner than this on a side is an icon, a
# badge or a tracking pixel.
POST_IMAGE_MIN_PX = _int("POST_IMAGE_MIN_PX", 200)
POST_IMAGE_MAX_MB = _int("POST_IMAGE_MAX_MB", 10)
# Wider than this is banner-SHAPED, which is a demotion rather than a veto —
# see POST_IMAGE_WIDE_PENALTY below for why it stopped being a hard drop.
POST_IMAGE_MAX_ASPECT = _float("POST_IMAGE_MAX_ASPECT", 4.0)
# CLIP layer: informative must BEAT decorative by a margin and clear a floor.
# "Closer to chart than to photo" is a weaker claim than "is a chart", which is
# why both conditions exist. The floor sits in the same ballpark as
# CONFIDENCE_THRESHOLD — the same CLIP cosine scale.
POST_IMAGE_MARGIN = _float("POST_IMAGE_MARGIN", 0.02)
POST_IMAGE_MIN_SCORE = _float("POST_IMAGE_MIN_SCORE", 0.22)
# An image before the first heading is nearly always cover art, so it faces a
# HIGHER floor rather than a hard drop — a rare post leads with its key chart,
# and vetoing position 0 would lose exactly the image most worth citing.
POST_IMAGE_HERO_PENALTY = _float("POST_IMAGE_HERO_PENALTY", 0.03)
# Banner-SHAPED gets the same treatment, and for a measured reason: on the
# first real seven-post corpus, every image the aspect rule vetoed outright
# was a genuine figure — a horizontal number line at 4.8:1, a
# Data->Model->Product flow diagram at 5.9:1 that was the most citable image
# in its post. Both score ~0.27 informative. Wide decorative strips are almost
# always thin in absolute terms too, so POST_IMAGE_MIN_PX still vetoes them.
POST_IMAGE_WIDE_PENALTY = _float("POST_IMAGE_WIDE_PENALTY", 0.03)
# Alt text that names a figure ("chart", "diagram", "table") relaxes the floor.
# A heuristic rather than a second embedding on purpose: scoring alt text in
# CLIP space would mix text-text and text-image cosines, which live on
# different scales — the calibration error RRF exists to avoid.
POST_IMAGE_ALT_BONUS = _float("POST_IMAGE_ALT_BONUS", 0.03)

# --- Presigned uploads (browser -> bucket, bypassing the API) -----------------
PRESIGN_EXPIRY_S = _int("PRESIGN_EXPIRY_S", 900)          # presigned PUT lifetime
PRESIGN_GET_EXPIRY_S = _int("PRESIGN_GET_EXPIRY_S", 3600)  # thumbnails / playback
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 2048)                # register rejects bigger objects
ALLOWED_UPLOAD_TYPES = ("video/",)                         # content-type must start with

# --- Video ingest lifecycle ---------------------------------------------------
# pending  = registered, waiting in our fair queue (not yet sent to Prefect)
# queued   = the dispatcher picked it and scheduled a Prefect run
# fetching = acquiring the source file; sampling = frames + dedup + thumbnails;
# embedding = CLIP + Qdrant upsert; skipped = duplicate (user_id, source_hash).
VIDEO_STATUSES = ("pending", "queued", "fetching", "sampling", "embedding",
                  "indexed", "skipped", "failed")

# --- Source manifest (multi-kind) ---------------------------------------------
# What a manifest row can describe. `deck` is still accepted by the schema ahead
# of its flow — the migration lands before anything depends on it. `post` is
# markdown: same document lifecycle as a paper, but cited by heading anchor
# instead of page number (migration 003).
SOURCE_KINDS = ("video", "paper", "deck", "post")

# The full lifecycle across kinds. Documents add two stages videos don't have:
# `parsing` (PDF -> per-page text) and `chunking` (pages -> page-aware chunks),
# which together sit where `sampling` sits for video.
SOURCE_STATUSES = ("pending", "queued", "fetching", "parsing", "chunking",
                   "sampling", "embedding", "indexed", "skipped", "failed")

# The vocabulary for the `stage` column — the last completed pipeline stage,
# which Epic 5's checkpointing reads to skip finished work on a resumed run.
SOURCE_STAGES = ("fetch", "parse", "chunk", "sample", "embed", "transcript")

# In-flight = occupying execution capacity (scheduled or running). This feeds
# DISPATCH_MAX_INFLIGHT accounting in the dispatcher, so `parsing` and
# `chunking` belong here from the start: a document in either state that didn't
# occupy a slot would let the dispatcher over-admit.
INFLIGHT_STATUSES = ("queued", "fetching", "parsing", "chunking", "sampling",
                     "embedding")

# Which kinds the dispatcher claims and routes to a flow. `deck` is absent on
# purpose: it has no flow yet, so pending decks must SIT pending rather than be
# claimed and crashed through a pipeline that can't handle them (the same
# hazard wfq_claim's kind filter originally closed for papers).
DISPATCH_KINDS = ("video", "paper", "post")

# --- Fair scheduling (WFQ) ----------------------------------------------------
# FIFO (default off): register enqueues to Prefect immediately -> Prefect runs
# them in submitted order, so one user with 50 videos blocks everyone behind
# them. Fair dispatch (WFQ, on): videos wait `pending` in Postgres and a
# dispatcher admits them round-robin ACROSS users, keeping only
# DISPATCH_MAX_INFLIGHT running at once — so the waiting line is fairly ordered
# in OUR DB, not FIFO inside Prefect. No user can starve the others.
ENABLE_FAIR_DISPATCH = _envbool("ENABLE_FAIR_DISPATCH", True)
# Max videos executing at once. Set to your total capacity:
# (worker machines) x WORKER_CONCURRENCY — anything above that would just pile
# up FIFO inside Prefect and defeat the fairness.
DISPATCH_MAX_INFLIGHT = _int("DISPATCH_MAX_INFLIGHT", _int("WORKER_CONCURRENCY", 2))
DISPATCH_INTERVAL_S = _float("DISPATCH_INTERVAL_S", 3.0)  # how often the dispatcher tops up

# --- Frame sampling (the biggest scaling lever) --------------------------------
# interval: one frame every FRAME_INTERVAL_SEC (widened to respect MAX_FRAMES).
# scene:    one frame per detected cut (ffmpeg scene filter).
FRAME_STRATEGY = os.getenv("FRAME_STRATEGY", "interval").strip().lower()
FRAME_INTERVAL_SEC = _float("FRAME_INTERVAL_SEC", 2.0)
SCENE_THRESHOLD = _float("SCENE_THRESHOLD", 0.4)
MAX_FRAMES = _int("MAX_FRAMES", 400)
THUMB_WIDTH = _int("THUMB_WIDTH", 480)   # frames are downscaled in the ffmpeg pass
THUMB_QUALITY = _int("THUMB_QUALITY", 3)  # ffmpeg -q:v (2 best .. 31 worst)

# Perceptual-hash dedup — drop visually-identical neighbours BEFORE embedding.
DEDUP_ENABLED = _envbool("DEDUP_ENABLED", True)
DEDUP_MAX_DISTANCE = _int("DEDUP_MAX_DISTANCE", 4)  # Hamming distance on 64-bit dHash

# --- CLIP embeddings ------------------------------------------------------------
# One model encodes frames and text queries into a shared space. Runs on CPU
# inside the worker today; EMBED_VERSION is stamped on every Qdrant point so a
# future re-embed (or an external GPU CLIP service) can replace stale vectors
# without guessing.
CLIP_MODEL = os.getenv("CLIP_MODEL", "clip-ViT-B-32").strip()
CLIP_BATCH = _int("CLIP_BATCH", 128)   # frames per embed call (inner batch is 32)
# Inference-service URL ("embedding is a URL"). Set -> api/worker send batches
# to the warm clip_service.py container instead of loading the model in-process
# (which costs each Prefect run subprocess a fresh ~15-30s torch load). Unset
# -> in-process embedding (simple mode, no extra service). Point it at a GPU
# machine later — nothing else changes.
CLIP_SERVICE_URL = os.getenv("CLIP_SERVICE_URL", "").strip().rstrip("/")
# Vector dimension override. 0 = auto: known CLIP models resolve from a table
# (so the API can create the collection at boot WITHOUT loading the model);
# unknown models load the model to measure. Set explicitly for custom models.
CLIP_DIM = _int("CLIP_DIM", 0)
EMBED_VERSION = os.getenv("EMBED_VERSION", f"{CLIP_MODEL}-v1")

# --- Multimodal: transcript (text) branch (Path 1) -----------------------------
# The visual branch is CLIP frames (above). This adds a SECOND branch: YouTube
# captions, chunked by time, embedded with a small semantic text model (bge via
# fastembed — CPU, free), in a separate Qdrant collection. At query time both
# branches run and fuse by RANK (RRF) — CLIP scores (~0.3) and text scores
# (~0.7) live on different scales, so raw-score comparison is meaningless.
# Uploaded files have no captions, so this is YouTube-only; a video with no
# captions just indexes visually (the branch is skipped, never fatal).
ENABLE_TRANSCRIPT = _envbool("ENABLE_TRANSCRIPT", True)
TEXT_COLLECTION = os.getenv("TEXT_COLLECTION", "moments_text")
# Transcript-branch embedding PROVIDER — env decides the model:
#   fastembed (default) -> bge via fastembed: CPU, free, NO API key (keeps search
#                          working keyless for a fresh cloner). Dim 384.
#   openai              -> OpenAI (or any OpenAI-compatible) embeddings API, e.g.
#                          text-embedding-3-small. Hosted, stronger retrieval,
#                          costs per call + needs a key. Reuses the LLM_* key /
#                          base_url by default (override with TEXT_EMBED_API_KEY /
#                          TEXT_EMBED_BASE_URL). Setting the provider alone flips
#                          the default model+dim to 3-small / 1536.
# The model & dim MUST match between indexing and querying, so switching provider
# means RE-SEEDING the transcript collection (its vector dim changes). The two
# branches fuse by RANK (RRF), so the text model is independent of CLIP.
TEXT_EMBED_PROVIDER = os.getenv("TEXT_EMBED_PROVIDER", "fastembed").strip().lower()
_TE_OPENAI = TEXT_EMBED_PROVIDER == "openai"
TEXT_EMBED_MODEL = os.getenv(
    "TEXT_EMBED_MODEL",
    "text-embedding-3-small" if _TE_OPENAI else "BAAI/bge-small-en-v1.5")
TEXT_EMBED_DIM = _int("TEXT_EMBED_DIM", 1536 if _TE_OPENAI else 384)
# openai provider: falls back to the LLM key/base_url in embeddings.py so ONE
# OpenAI key can power both the answer and the text embeddings.
TEXT_EMBED_API_KEY = os.getenv("TEXT_EMBED_API_KEY", "").strip()
TEXT_EMBED_BASE_URL = os.getenv("TEXT_EMBED_BASE_URL", "").strip()
TEXT_EMBED_VERSION = os.getenv("TEXT_EMBED_VERSION", f"{TEXT_EMBED_MODEL}-v1")
# Transcript chunking: group caption cues into ~CHUNK_SECONDS windows so a chunk
# is a coherent spoken passage with a real t_start/t_end, not one tiny cue.
TRANSCRIPT_CHUNK_SECONDS = _float("TRANSCRIPT_CHUNK_SECONDS", 20.0)
TRANSCRIPT_LANGS = [c.strip() for c in
                    os.getenv("TRANSCRIPT_LANGS", "en,en-US,en-GB").split(",") if c.strip()]

# --- Fusion (multimodal retrieval) ---------------------------------------------
# RRF: rank-based fusion across branches (score-agnostic). rrf = 1/(K + rank).
RRF_K = _int("RRF_K", 60)
# Hits from either branch within this many seconds are the SAME moment.
FUSION_WINDOW_S = _float("FUSION_WINDOW_S", 15.0)
# When a window has BOTH a frame and a transcript hit, multiply its score —
# two independent modalities agreeing is the strongest relevance signal.
CROSS_MODAL_BOOST = _float("CROSS_MODAL_BOOST", 1.5)
# Per-branch candidates fetched before fusion.
BRANCH_TOP_K = _int("BRANCH_TOP_K", 20)

# --- YouTube download hardening ---------------------------------------------------
# YouTube increasingly answers yt-dlp's default web client with "Sign in to
# confirm you're not a bot". Mitigations, in order of reliability:
#   YT_COOKIES_FILE  path to a Netscape cookies.txt exported from a logged-in
#                    browser (yt-dlp wiki: "Exporting YouTube cookies").
#                    In docker-compose, drop it at ./data/cookies.txt and set
#                    YT_COOKIES_FILE=/app/data/cookies.txt
#   YT_PROXY_URL     route YouTube traffic through a (residential) proxy,
#                    e.g. http://user:pass@host:port
#   YT_RETRY_CLIENTS on a bot-check error, automatically retry once with these
#                    alternate YouTube player clients (comma-separated).
# Cookies make yt-dlp an authenticated client — the one fix that works from
# ANY IP (home OR datacenter). Two ways to supply them, so the same code works
# local and deployed:
#   YT_COOKIES_FILE  path to a mounted cookies.txt   (easy locally)
#   YT_COOKIES_B64   base64 of cookies.txt as a secret (Fly/cloud: no file mount
#                    needed — the worker writes it to a temp file at runtime)
YT_COOKIES_FILE = os.getenv("YT_COOKIES_FILE", "").strip()
YT_COOKIES_B64 = os.getenv("YT_COOKIES_B64", "").strip()
YT_PROXY_URL = os.getenv("YT_PROXY_URL", "").strip()
# Player clients. Default EMPTY = let yt-dlp pick (best, once a JS runtime is
# present — see below). Forcing tv/android used to help pre-JS-runtime, but now
# those clients hand back media URLs that 403 on download, so we only fall back
# to them if the default attempt fails outright.
YT_PLAYER_CLIENTS = [c.strip() for c in
                     os.getenv("YT_PLAYER_CLIENTS", "").split(",") if c.strip()]
YT_FALLBACK_CLIENTS = [c.strip() for c in
                       os.getenv("YT_FALLBACK_CLIENTS", "tv,android,ios").split(",") if c.strip()]
# yt-dlp 2025+ needs a JavaScript runtime to compute YouTube signatures, plus
# its EJS challenge-solver component — WITHOUT these every video fails with
# "This video is not available" / "Requested format is not available". The
# Dockerfile installs Node; these tell yt-dlp to use it + fetch the solver.
# (For bare-process dev, install node or deno yourself.)
YT_JS_RUNTIMES = [c.strip() for c in
                  os.getenv("YT_JS_RUNTIMES", "node").split(",") if c.strip()]
YT_REMOTE_COMPONENTS = [c.strip() for c in
                        os.getenv("YT_REMOTE_COMPONENTS", "ejs:github").split(",") if c.strip()]

# --- Sample corpus ------------------------------------------------------------
# On boot the worker auto-ingests the four "Deep Dive into LLMs" sample talks
# (src/samples.py) if they aren't indexed yet — a fresh clone is queryable on
# the / page without running anything by hand. Set false to skip.
SEED_SAMPLE_VIDEOS = _envbool("SEED_SAMPLE_VIDEOS", True)

# --- Work orchestration (Prefect Cloud) ----------------------------------------
# The SDK reads PREFECT_API_URL / PREFECT_API_KEY from the environment directly.
# WORKER_CONCURRENCY is read by worker.py; retries live on the flow's tasks.

# --- Qdrant ----------------------------------------------------------------------
# One shared multi-tenant collection: every point carries user_id (tenant payload
# index) and every search/upsert/delete is user_id-filtered.
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip() or os.getenv("QDRANT_TOKEN", "").strip()
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", str(DATA / "qdrant"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "moments")
# Low-RAM profile: original vectors on disk, int8-quantized copies pinned in
# RAM (~4x smaller), HNSW graph on disk; queries rescore against the originals.
# Frames balloon vector counts fast, so these default ON.
QDRANT_QUANTIZATION = _envbool("QDRANT_QUANTIZATION", True)
QDRANT_ON_DISK = _envbool("QDRANT_ON_DISK", True)
QDRANT_HNSW_ON_DISK = _envbool("QDRANT_HNSW_ON_DISK", True)

# --- Retrieval / faithfulness ------------------------------------------------------
TOP_K = _int("TOP_K", 6)                 # frames fed to the multimodal LLM (3-8)
KNN_K = _int("KNN_K", 24)                # candidates fetched before trimming to TOP_K
# Gate 1: abstain WITHOUT calling the LLM if BOTH branches' best raw score is
# below their threshold. Fusion scores are RRF (tiny), so the gate uses each
# branch's own raw cosine. CLIP text->image cosines run low (~0.2-0.35); bge
# text-text cosines run higher (~0.5-0.7 for real matches). 0 disables.
CONFIDENCE_THRESHOLD = _float("CONFIDENCE_THRESHOLD", 0.2)              # visual (CLIP)
TEXT_CONFIDENCE_THRESHOLD = _float("TEXT_CONFIDENCE_THRESHOLD", 0.35)  # transcript (bge)

# --- Multimodal LLM (answer synthesis only — retrieval works without it) -----------
# LLM_PROVIDER: openai | nvidia | anthropic ("openai" also covers any
# OpenAI-compatible server via LLM_BASE_URL: Ollama, vLLM, OpenRouter, ...).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 1024)
LLM_IMAGE_MAX_PX = _int("LLM_IMAGE_MAX_PX", 512)  # frames are downscaled again before the LLM


def llm_configured() -> bool:
    # Local OpenAI-compatible servers often need no key, so a base_url alone counts.
    return bool(LLM_API_KEY or LLM_BASE_URL)
