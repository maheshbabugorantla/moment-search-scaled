"""Acquiring and parsing a markdown post, cited by heading anchor.

The third source kind, and the only one whose locator is *semantic*. A paper
chunk cites `page 7` — a number the renderer's margins invented, which no URL
can express. A post chunk cites `#what-breaks-first` — the slug the author's
own heading produced, which survives a re-render, survives edits elsewhere in
the document, and appends to the post URL to scroll a reader straight to the
passage. That is the entire reason this kind exists instead of rendering posts
to PDF and calling them papers.

Three stages live here; chunking lives in rag/chunk.py (`chunk_markdown`):

  fetch   stream the URI to scratch + a content-addressed object, exactly as
          paper.fetch_paper does (shared streamer in ingest/download.py)
  read    bytes -> text, with the guards markdown needs instead of magic bytes
  parse   text -> Sections carrying (anchor, heading path, paragraphs, images)

Debug entrypoint:
    python -m src.ingest.post <uri> [user_id]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .. import storage
from ..config import MAX_POST_MB, POST_KEY_PREFIX
from .download import stream_to_file
from .errors import SourceError
from .fetch import scratch_dir, sha256_file
from .textnorm import normalize_unicode

# ── Fetch ─────────────────────────────────────────────────────────────────────


def post_key(user_id: str, content_hash: str) -> str:
    """Durable object key — user-scoped like every other prefix, content-
    addressed within the tenant, so one post registered twice is one object."""
    return f"{POST_KEY_PREFIX}{user_id}/{content_hash}.md"


def check_not_html(first_bytes: bytes) -> None:
    """Markdown has no magic bytes, so the cheap guard is the opposite one: is
    this obviously a saved web page rather than an export?

    Prefix-only, and only on the leading non-whitespace. Substack posts embed
    `<div>`, `<figure>` and `<img>` mid-body all the time — a check for
    "contains HTML" would reject the real thing.
    """
    lead = first_bytes.lstrip()[:64].lower()
    if lead.startswith(b"<!doctype") or lead.startswith(b"<html"):
        raise SourceError(
            "got HTML, not markdown — export the post as .md "
            "(a saved web page carries the site chrome, not the article)")


def read_markdown(path: Path) -> str:
    """Downloaded bytes -> text, or a readable reason why not.

    UTF-8 is validated on the WHOLE file rather than per chunk: a multi-byte
    character straddling a read boundary would fail a per-chunk decode on a
    perfectly valid document, and em dashes and curly quotes are everywhere in
    the posts this kind exists for. At MAX_POST_MB the whole file fits in
    memory comfortably.
    """
    raw = path.read_bytes()
    check_not_html(raw)
    if b"\x00" in raw:
        raise SourceError(
            "the content contains NUL bytes — that is a binary file, not markdown")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceError(
            f"not valid UTF-8 (byte {exc.start}) — re-export the post as "
            "UTF-8 markdown") from exc
    if not text.strip():
        raise SourceError("the markdown file is empty")
    # The same normalization papers get, MINUS dehyphenate/unwrap_lines: those
    # repair PDF line-breaking, and markdown's newlines are structure we parse.
    # This still folds curly quotes to ASCII and drops the BOM, so a post and a
    # paper quoting the same sentence embed as the same sentence.
    return normalize_unicode(text)


def fetch_post(uri: str, user_id: str, doc_id: str) -> dict:
    """URI -> scratch file + durable content-addressed object.

    Returns {"storage_key", "content_hash", "byte_size", "scratch_path"}. The
    scratch file is the caller's to delete (the flow's `finally`); the durable
    object stays.
    """
    dest = scratch_dir() / f"{doc_id}.md"
    if uri.startswith("storage://"):
        storage.download_to(uri[len("storage://"):], dest)
        size = dest.stat().st_size
        if size > MAX_POST_MB * (1 << 20):
            raise SourceError(
                f"markdown exceeds the {MAX_POST_MB} MB limit "
                f"({size / (1 << 20):.0f} MB)")
        with dest.open("rb") as fh:
            check_not_html(fh.read(1024))
        content_hash = sha256_file(dest)
    else:
        content_hash, size = stream_to_file(
            uri, dest, max_mb=MAX_POST_MB, what="markdown",
            first_chunk_check=check_not_html)

    key = post_key(user_id, content_hash)
    if not storage.exists(key):  # same content already stored -> no re-upload
        storage.upload_file(dest, key, content_type="text/markdown")
    return {"storage_key": key, "content_hash": content_hash,
            "byte_size": size, "scratch_path": str(dest)}


# ── Parse ─────────────────────────────────────────────────────────────────────

TOP_ANCHOR = "_top"  # content before the first heading

# YAML front matter, as every markdown exporter writes it: a --- fenced block
# at the very top of the file. Substack/blog exporters put the canonical post
# URL in there, which is the one piece of metadata this kind genuinely needs —
# the citation deeplink is `{that url}#{anchor}`, and the registered URI may be
# a storage:// key that no reader can follow.
_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n",
                          re.DOTALL)
_YAML_LINE = re.compile(r"^([A-Za-z][\w.-]*)\s*:\s*(.*)$")

_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# A pseudo-heading: a bold line ALONE on its line. Strict by design — Substack
# authors who never type `#` write sections this way, but mid-paragraph bold is
# constant, and a loose match would shred a post into dozens of bogus sections.
_BOLD_LINE = re.compile(r"^\*\*([^*]+?)\*\*[:.]?$")
_FENCE = re.compile(r"^\s*(?:```|~~~)")
_HR = re.compile(r"^\s*([-*_])\s*(?:\1\s*){2,}$")
_QUOTE = re.compile(r"^\s{0,3}>\s?")

_IMAGE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)")
_HTML_IMAGE = re.compile(r"<img\b[^>]*?>", re.IGNORECASE)
_HTML_ATTR = re.compile(r"""(\w+)\s*=\s*["']([^"']*)["']""")
_LINK = re.compile(r"\[([^\]]*)\]\(\s*<?[^)\s>]+>?(?:\s+[\"'][^\"']*[\"'])?\s*\)")
_FOOTNOTE = re.compile(r"\[\^[^\]]+\]")
_BOLD = re.compile(r"\*\*([^*]+?)\*\*|__([^_]+?)__")
_INLINE_CODE = re.compile(r"`([^`]+)`")
# GitHub's slug: lowercase, punctuation dropped, each space becomes a dash.
_SLUG_DROP = re.compile(r"[^\w\- ]", re.UNICODE)


@dataclass(frozen=True)
class ImageRef:
    url: str
    alt: str
    anchor: str            # the section the image sits in
    position: int          # 0-based index in the document's image sequence
    before_first_heading: bool  # cover/hero art candidate (REC-337)


@dataclass(frozen=True)
class Section:
    anchor: str
    heading: str           # the trail: "Scaling laws > What breaks first"
    level: int
    anchor_native: bool    # False -> a synthesised anchor that won't scroll
    paragraphs: tuple[str, ...] = ()
    images: tuple[ImageRef, ...] = field(default=())


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """(metadata, body) — or ({}, text) when there is no front matter.

    Parsed with a flat scanner rather than a YAML library on purpose: the
    fields worth having are scalars (title, url, author, date), a real parser
    would be a new dependency for that, and a malformed block must degrade to
    "no metadata" rather than fail an ingest. Nested structures and list
    syntax are simply not read.

    Without this, every exported post indexes `title:`, `author:` and `tags:`
    as prose in its very first chunk — the chunk most likely to be retrieved
    for a query about what the post is.
    """
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        field = _YAML_LINE.match(line)
        if not field:
            continue  # a list item, a continuation, a comment — not a scalar
        value = field.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            meta[field.group(1).lower()] = value
    return meta, text[match.end():]


def strip_inline(text: str) -> str:
    """Markdown inline syntax -> the words a reader sees.

    Links reduce to their visible text (the URL is noise to an embedding
    model), bold/inline-code markers go, footnote markers go. Images are NOT
    handled here — they are lifted out earlier, with their alt text.
    """
    text = _FOOTNOTE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _BOLD.sub(lambda m: m.group(1) or m.group(2), text)
    text = _INLINE_CODE.sub(r"\1", text)
    return text


def slugify(text: str) -> str:
    """GitHub-style heading slug — the algorithm Substack and GitHub renderers
    both use, so `{post_url}#{slug}` scrolls the live page."""
    s = strip_inline(text).strip().lower()
    s = _SLUG_DROP.sub("", s)
    return s.replace(" ", "-")


_MAX_SYNTH_SLUG = 60


def _trim_slug(slug: str) -> str:
    """Shorten a synthesised slug at a word boundary. A bold pseudo-heading is
    often a whole sentence, and a 150-character anchor in every payload is
    noise with no upside — nothing will ever resolve it."""
    if len(slug) <= _MAX_SYNTH_SLUG:
        return slug
    cut = slug.rfind("-", 0, _MAX_SYNTH_SLUG)
    return slug[:cut] if cut > _MAX_SYNTH_SLUG // 2 else slug[:_MAX_SYNTH_SLUG]


def _html_image(tag: str) -> tuple[str, str] | None:
    attrs = {k.lower(): v for k, v in _HTML_ATTR.findall(tag)}
    src = attrs.get("src", "").strip()
    return (src, attrs.get("alt", "").strip()) if src else None


def _paragraphs_of(lines: list[str]) -> tuple[str, ...]:
    """Section lines -> paragraphs. Blank lines separate; a fenced code block
    is one paragraph however many blank lines it contains, because splitting a
    fence produces two chunks that are each invalid code."""
    paras: list[str] = []
    buf: list[str] = []
    in_fence = False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        if in_fence:
            buf.append(line)
            continue
        if not line.strip() or _HR.match(line):
            if buf:
                paras.append("\n".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        paras.append("\n".join(buf))
    return tuple(p for p in (x.strip() for x in paras) if p)


def parse_markdown(text: str, *, title: str | None = None) -> list[Section]:
    """Markdown -> sections carrying an anchor, a heading path, paragraphs and
    image refs.

    Splits on ATX headings (`#`..`######`) plus the bold-line Substack-ism.
    Content before the first heading becomes the `_top` pseudo-section, which
    every renderer resolves (it is the page itself).

    Duplicate headings get `-1`, `-2` suffixes — and the FIRST occurrence stays
    bare, exactly as GitHub numbers them. Suffixing the first would break the
    deeplink for the common case, which is the whole point of the kind.

    YAML front matter is stripped before anything else; use split_frontmatter()
    directly when you also want the metadata.
    """
    _, text = split_frontmatter(text)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    sections: list[Section] = []
    seen: Counter[str] = Counter()
    trail: list[tuple[int, str]] = []   # (level, heading text) — the open path
    cur_anchor = TOP_ANCHOR
    cur_heading = (title or "").strip()
    cur_level = 0
    cur_native = True   # `_top` resolves on every renderer
    buf: list[str] = []
    pending: list[tuple[str, str]] = []  # (url, alt) seen in the current section
    in_fence = False
    seen_heading = False

    def _flush() -> None:
        nonlocal buf, pending
        paragraphs = _paragraphs_of(buf)
        refs = tuple(
            ImageRef(url=url, alt=alt, anchor=cur_anchor,
                     position=pos, before_first_heading=not seen_heading)
            for pos, (url, alt) in enumerate(pending))
        if paragraphs or refs:
            sections.append(Section(anchor=cur_anchor, heading=cur_heading,
                                    level=cur_level, anchor_native=cur_native,
                                    paragraphs=paragraphs, images=refs))
        buf, pending = [], []

    def _anchor_for(heading_text: str, native: bool) -> str:
        base = slugify(heading_text) or f"section-{len(sections) + 1}"
        if not native:
            # A synthesised anchor never has to match a renderer's, so it can be
            # trimmed to something readable. Native ones are NOT trimmed at any
            # length: they must stay byte-identical to what the renderer emits
            # or `{url}#{anchor}` silently stops scrolling.
            base = _trim_slug(base)
        seen[base] += 1
        return base if seen[base] == 1 else f"{base}-{seen[base] - 1}"

    def _open(level: int, heading_text: str, native: bool) -> None:
        nonlocal cur_anchor, cur_heading, cur_level, cur_native, trail
        while trail and trail[-1][0] >= level:
            trail.pop()
        trail.append((level, heading_text))
        cur_anchor = _anchor_for(heading_text, native)
        cur_heading = " > ".join(h for _, h in trail)
        cur_level = level
        cur_native = native

    for raw in lines:
        if _FENCE.match(raw):
            in_fence = not in_fence
            buf.append(raw)
            continue
        if in_fence:
            buf.append(raw)      # verbatim: code fences are often the substance
            continue

        atx = _ATX.match(raw)
        if atx:
            _flush()
            seen_heading = True
            _open(len(atx.group(1)), strip_inline(atx.group(2)).strip(), True)
            continue

        stripped = _QUOTE.sub("", raw).strip()
        bold = _BOLD_LINE.match(stripped)
        if bold:
            _flush()
            # One level below whatever is open, so the trail stays sensible.
            # Flagged non-native: this anchor is ours, not the renderer's, so
            # `{url}#{anchor}` will NOT scroll — the payload says so and Epic 4
            # falls back to the nearest native anchor.
            _open(min(cur_level + 1, 6), strip_inline(bold.group(1)).strip(), False)
            continue

        line = _QUOTE.sub("", raw)
        for m in _IMAGE.finditer(line):
            pending.append((m.group(2), strip_inline(m.group(1)).strip()))
        line = _IMAGE.sub("", line)
        for m in _HTML_IMAGE.finditer(line):
            found = _html_image(m.group(0))
            if found:
                pending.append(found)
        line = _HTML_IMAGE.sub("", line)
        buf.append(strip_inline(line))
    _flush()

    # `position` is assigned per section above from the section-local index;
    # renumber across the document so REC-337's position-0 hero rule means what
    # it says. Rebuilt rather than mutated — the dataclasses are frozen.
    numbered: list[Section] = []
    pos = 0
    for section in sections:
        refs = []
        for ref in section.images:
            refs.append(ImageRef(url=ref.url, alt=ref.alt, anchor=ref.anchor,
                                 position=pos,
                                 before_first_heading=ref.before_first_heading))
            pos += 1
        numbered.append(Section(anchor=section.anchor, heading=section.heading,
                                level=section.level,
                                anchor_native=section.anchor_native,
                                paragraphs=section.paragraphs,
                                images=tuple(refs)))
    return numbered


if __name__ == "__main__":  # python -m src.ingest.post <uri> [user_id]
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.ingest.post <uri> [user_id]")
    _uri = sys.argv[1]
    _user = sys.argv[2] if len(sys.argv) > 2 else "default"
    _handle = fetch_post(_uri, _user, "doc_debug")
    _path = Path(_handle.pop("scratch_path"))
    print(json.dumps(_handle, indent=2))

    from ..rag.chunk import chunk_markdown

    _sections = parse_markdown(read_markdown(_path))
    _chunks = chunk_markdown(_sections)
    for _s in _sections:
        _n = sum(1 for c in _chunks if c.anchor == _s.anchor)
        _native = "native" if _s.anchor_native else "SYNTHESISED"
        print(f"  #{_s.anchor:<40} {_native:<12} {_n} chunk(s), "
              f"{len(_s.images)} image(s)  — {_s.heading}")
    _path.unlink(missing_ok=True)
