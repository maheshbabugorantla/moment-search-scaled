"""Deciding which of a post's images is worth citing.

The requirement is a *gate*, not image indexing for its own sake. A post's
chart deserves the cross-modal treatment a deck slide's caption gets — but
Substack posts are full of hero art, section dividers, author avatars and
memes, and a decorative image must never end up as a citation. False positives
here are worse than false negatives: an answer illustrated by the author's
headshot is a worse answer than one with no picture.

Two layers, cheapest first:

  1. **Heuristics** — free, deterministic, and they kill most of the junk.
     Tiny images are icons and tracking pixels; 4:1 strips are banners and
     dividers; animated GIFs are reaction images; some URLs announce
     themselves. These are hard drops.

  2. **CLIP zero-shot** — the warm clip service already runs, so this costs no
     new model, no API key and no per-image fee. The image is embedded once
     and scored against two prompt banks; informative has to beat decorative
     by a margin AND clear a floor, because "closer to chart than to photo"
     is not the same claim as "is a chart".

Two decisions worth stating, because both are judgement calls the spec left
open:

* **Position 0 demotes, it does not veto.** An image before the first heading
  is nearly always cover art — but a rare post leads with its key chart, and
  a hard drop would lose exactly the image most worth citing. So a hero image
  faces a HIGHER CLIP floor instead of a closed door.

* **Alt text is a heuristic, not a score.** The obvious move — embed the alt
  text and add its similarity to the informative side — mixes CLIP text-text
  cosines with text-image cosines, which live on different scales; the
  transcript branch exists precisely because CLIP's text encoder is tuned to
  match images, not other text (see rag/embeddings.py). So alt text instead
  relaxes the floor by a fixed amount when it names what the image is
  ("figure", "chart", "diagram"). Deterministic, explainable in a log line,
  and no invented calibration.

Everything here is best-effort: an image that fails to fetch or decode is
logged and skipped, never raised. The parse stage has no retries, so one dead
image URL must not fail a whole post — the same rule the transcript branch
follows.
"""
from __future__ import annotations

import io
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urljoin

import numpy as np

from ..config import (POST_IMAGE_ALT_BONUS, POST_IMAGE_HERO_PENALTY,
                      POST_IMAGE_MARGIN, POST_IMAGE_MAX_ASPECT,
                      POST_IMAGE_MAX_MB, POST_IMAGE_MIN_PX,
                      POST_IMAGE_MIN_SCORE, THUMB_WIDTH)

_TIMEOUT_S = 20

# Scored against the image in CLIP's shared space. Short, concrete noun
# phrases: CLIP's text encoder was trained on captions, and it reads
# "a chart or graph of data" far better than "informative".
INFORMATIVE_PROMPTS = (
    "a chart or graph of data",
    "a technical diagram or architecture illustration",
    "a screenshot of software or code",
    "a table of numbers",
)
DECORATIVE_PROMPTS = (
    "an artistic photograph",
    "an abstract decorative illustration",
    "a stock photo of a person",
    "a meme or joke image",
)

# Words an author uses when they are labelling something worth looking at.
_ALT_INFORMATIVE = re.compile(
    r"\b(chart|graph|plot|figure|fig|diagram|table|screenshot|architecture|"
    r"benchmark|curve|distribution|schematic|flow)\b", re.IGNORECASE)

# URLs that announce what they are before a byte is downloaded.
_URL_SMELLS = re.compile(
    r"(/avatar|/avatars/|gravatar|/profile[-_/]|/icon[-_s]?/|/logo[-_s]?/|"
    r"/badge|shields\.io|/emoji/|/spacer|/pixel|/tracking|/divider|"
    r"twitter_card|/og[-_]image|/social[-_]card|/share[-_]card)",
    re.IGNORECASE)


@dataclass(frozen=True)
class Verdict:
    keep: bool
    reason: str            # which rule decided, for the log and the audit trail
    img_class: str = ""    # "informative" | "decorative" | "" when never scored
    img_score: float = 0.0  # the winning informative cosine
    jpeg: bytes = b""      # downscaled JPEG, only when keep
    # The CLIP vector the verdict was reached with, handed back so the caller
    # upserts what was actually classified instead of re-embedding the same
    # bytes. compare=False because array equality is not a bool.
    vector: "np.ndarray | None" = field(default=None, compare=False)


def url_is_junk(url: str) -> str:
    """A drop reason from the URL alone, or "" to keep looking. Runs before any
    download, so the obvious junk costs nothing."""
    if _URL_SMELLS.search(url):
        return "url looks like chrome (avatar/icon/badge/share-card)"
    return ""


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "momentsearch/1.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return resp.read(POST_IMAGE_MAX_MB * (1 << 20) + 1)


def shape_is_junk(width: int, height: int, animated: bool) -> str:
    """A drop reason from the decoded image's shape, or ""."""
    if animated:
        return "animated GIF"
    if min(width, height) < POST_IMAGE_MIN_PX:
        return f"too small ({width}x{height}, under {POST_IMAGE_MIN_PX}px)"
    ratio = max(width, height) / max(1, min(width, height))
    if ratio > POST_IMAGE_MAX_ASPECT:
        return f"banner-shaped ({width}x{height}, {ratio:.1f}:1)"
    return ""


def classify(vector: np.ndarray, prompt_vectors: dict[str, np.ndarray], *,
             hero: bool, alt: str) -> tuple[bool, str, float]:
    """CLIP verdict for one already-embedded image.

    Returns (keep, img_class, informative_score). Vectors are L2-normalized by
    embeddings.py, so the dot product IS the cosine.

    Split out from the fetch/decode path with no I/O of its own, so the margin
    logic is testable with stubbed vectors — the part most likely to be tuned.
    """
    info = float(max(vector @ prompt_vectors[p] for p in INFORMATIVE_PROMPTS))
    deco = float(max(vector @ prompt_vectors[p] for p in DECORATIVE_PROMPTS))

    floor = POST_IMAGE_MIN_SCORE
    if hero:
        floor += POST_IMAGE_HERO_PENALTY   # demoted, not vetoed
    if _ALT_INFORMATIVE.search(alt or ""):
        floor -= POST_IMAGE_ALT_BONUS      # the author named it a figure

    keep = info >= deco + POST_IMAGE_MARGIN and info >= floor
    return keep, ("informative" if keep else "decorative"), info


def _to_jpeg(raw: bytes) -> tuple[bytes, int, int, bool]:
    """Decode, note the shape, downscale to THUMB_WIDTH, re-encode as JPEG —
    the same format and width every citation thumbnail already uses."""
    from PIL import Image

    img = Image.open(io.BytesIO(raw))
    animated = bool(getattr(img, "is_animated", False))
    width, height = img.size
    img = img.convert("RGB")
    if width > THUMB_WIDTH:
        img = img.resize((THUMB_WIDTH, max(1, round(height * THUMB_WIDTH / width))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img.close()
    return buf.getvalue(), width, height, animated


def prompt_bank() -> dict[str, np.ndarray]:
    """Embed both prompt banks once per post, not once per image."""
    from ..rag.embeddings import embed_text

    return {p: embed_text(p)
            for p in INFORMATIVE_PROMPTS + DECORATIVE_PROMPTS}


def resolve(url: str, base_uri: str = "") -> str:
    """Exported markdown routinely references images relatively
    ("assets/chart.png"), so resolve against the post's own URI the way the
    renderer would. Absolute URLs pass through untouched."""
    return urljoin(base_uri, url) if base_uri else url


def judge(url: str, *, base_uri: str = "", alt: str = "", hero: bool = False,
          prompts: dict[str, np.ndarray] | None = None) -> Verdict:
    """One image ref -> a keep/drop verdict with the reason that decided it.

    Never raises: a fetch failure, a decode failure or an unreachable clip
    service returns a drop verdict carrying the reason. The parse stage has no
    retries, so one bad image URL must not cost a whole post.
    """
    url = resolve(url, base_uri)
    smell = url_is_junk(url)
    if smell:
        return Verdict(keep=False, reason=smell)

    try:
        raw = _fetch(url)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Verdict(keep=False, reason=f"fetch failed: {type(exc).__name__}")
    if len(raw) > POST_IMAGE_MAX_MB * (1 << 20):
        return Verdict(keep=False, reason=f"over {POST_IMAGE_MAX_MB} MB")

    try:
        jpeg, width, height, animated = _to_jpeg(raw)
    except Exception as exc:  # Pillow raises a wide family on bad bytes
        return Verdict(keep=False, reason=f"undecodable: {type(exc).__name__}")

    shape = shape_is_junk(width, height, animated)
    if shape:
        return Verdict(keep=False, reason=shape)

    try:
        from ..rag.embeddings import embed_jpegs

        prompts = prompts if prompts is not None else prompt_bank()
        vector = embed_jpegs([jpeg])[0]
    except Exception as exc:
        return Verdict(keep=False, reason=f"classifier unavailable: {type(exc).__name__}")

    keep, img_class, score = classify(vector, prompts, hero=hero, alt=alt)
    reason = (f"{img_class} ({score:.3f})" if not hero
              else f"{img_class} ({score:.3f}, hero floor)")
    return Verdict(keep=keep, reason=reason, img_class=img_class,
                   img_score=score, jpeg=jpeg if keep else b"",
                   vector=vector if keep else None)
