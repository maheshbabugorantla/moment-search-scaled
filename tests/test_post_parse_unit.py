"""Unit tests for markdown parsing, anchors and section-bounded chunking.

The invariant a citation will stand on: every chunk carries an anchor that
`{post_url}#{anchor}` can actually scroll to, and the same markdown always
yields the same anchors in the same order — chunk idx decides the Qdrant point
id, so a redelivered ingest must overwrite rather than duplicate.

These import src directly (the tests container sets PYTHONPATH=/app): the
guards under test are pure logic, provable without a network or a bucket.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest.post import (TOP_ANCHOR, parse_markdown, post_key,
                             read_markdown, slugify, split_frontmatter,
                             strip_inline)
from src.ingest.errors import SourceError
from src.rag.chunk import chunk_markdown


# ── Front matter: metadata, not prose ────────────────────────────────────────

_EXPORT = '''---
title: The Rise of the AI Engineer
subtitle: "Emergent capabilities are creating an emerging title"
author: Latent.Space
date: 2023-06-30
url: "https://www.latent.space/p/ai-engineer"
tags: [ai-engineering, llms]
---

# The Rise of the AI Engineer

The body starts here.
'''


def test_front_matter_is_read_as_metadata() -> None:
    meta, body = split_frontmatter(_EXPORT)
    assert meta["title"] == "The Rise of the AI Engineer"
    assert meta["url"] == "https://www.latent.space/p/ai-engineer"   # unquoted
    assert meta["author"] == "Latent.Space"
    assert body.lstrip().startswith("# The Rise")


def test_front_matter_never_reaches_a_chunk() -> None:
    """Without this, every exported post indexes `title:`, `author:` and
    `tags:` as prose in its FIRST chunk — the chunk most likely to be
    retrieved for a question about what the post is."""
    text = " ".join(p for s in parse_markdown(_EXPORT) for p in s.paragraphs)
    for leaked in ("title:", "author:", "date:", "tags:", "subtitle:"):
        assert leaked not in text
    assert "The body starts here." in text


def test_a_document_without_front_matter_is_untouched() -> None:
    meta, body = split_frontmatter("# Title\n\nbody\n")
    assert meta == {} and body == "# Title\n\nbody\n"


def test_a_horizontal_rule_mid_document_is_not_mistaken_for_front_matter() -> None:
    """`---` is also a horizontal rule; only a block at the very TOP counts."""
    meta, body = split_frontmatter("intro\n\n---\n\nmore\n")
    assert meta == {} and body.startswith("intro")


def test_malformed_front_matter_degrades_to_no_metadata() -> None:
    """A broken block must cost the metadata, never the ingest."""
    meta, body = split_frontmatter("---\nnot: [a, valid\n  - yaml\n---\nbody\n")
    assert "not" in meta            # the scalar-ish line is read
    assert "- yaml" not in str(meta)  # the list continuation is skipped
    assert body.strip() == "body"


# ── Anchors: the whole reason this kind exists ───────────────────────────────

@pytest.mark.parametrize("heading,expected", [
    ("Scaling laws", "scaling-laws"),
    ("What breaks first?", "what-breaks-first"),
    ("The 10x engineer", "the-10x-engineer"),
    ("Don't panic", "dont-panic"),
    ("Notes & caveats", "notes--caveats"),   # the space around & survives as --
    ("**Bold heading**", "bold-heading"),
    ("`code` in a heading", "code-in-a-heading"),
    ("A [linked](https://x.com) word", "a-linked-word"),
])
def test_slugify_matches_the_github_algorithm(heading: str, expected: str) -> None:
    assert slugify(heading) == expected


def test_the_first_duplicate_heading_keeps_the_bare_anchor() -> None:
    """GitHub numbers from the SECOND occurrence. Suffixing the first would
    break the deeplink for the common case — which is the whole point."""
    sections = parse_markdown(
        "## Notes\nalpha\n\n## Notes\nbeta\n\n## Notes\ngamma\n")
    assert [s.anchor for s in sections] == ["notes", "notes-1", "notes-2"]


def test_content_before_the_first_heading_gets_the_top_anchor() -> None:
    sections = parse_markdown("An opening paragraph.\n\n## Later\nmore\n")
    assert sections[0].anchor == TOP_ANCHOR
    assert sections[0].anchor_native is True
    assert "opening paragraph" in sections[0].paragraphs[0]


def test_a_document_that_opens_with_its_heading_has_no_empty_top_section() -> None:
    sections = parse_markdown("# Title\n\nbody text\n")
    assert [s.anchor for s in sections] == ["title"]


def test_a_bold_line_becomes_a_pseudo_heading_flagged_non_native() -> None:
    """The one Substack-ism worth special-casing: authors who never type `#`."""
    sections = parse_markdown(
        "## Real heading\nintro\n\n**A bold section**\n\nits body\n")
    by_anchor = {s.anchor: s for s in sections}
    assert by_anchor["real-heading"].anchor_native is True
    assert by_anchor["a-bold-section"].anchor_native is False
    assert by_anchor["a-bold-section"].level == 3  # one below the h2


def test_a_synthesised_anchor_is_trimmed_but_a_native_one_never_is() -> None:
    """A bold pseudo-heading is often a whole sentence, and its anchor resolves
    nowhere, so it is trimmed. A native anchor must stay byte-identical to what
    the renderer emits at ANY length or the deeplink silently stops working."""
    sentence = " ".join(["word"] * 30)
    sections = parse_markdown(f"## {sentence}\nx\n\n**{sentence}**\n\ny\n")
    native, synth = sections[0], sections[1]
    assert native.anchor == slugify(sentence)      # untrimmed, however long
    assert len(synth.anchor) <= 60
    assert synth.anchor_native is False


def test_mid_paragraph_bold_does_not_split_a_section() -> None:
    """A loose match here would shred a real post into dozens of sections."""
    md = ("## One\n"
          "A sentence with **bold words** inside it, and another **here** too.\n"
          "Still the same paragraph **and more**.\n")
    sections = parse_markdown(md)
    assert [s.anchor for s in sections] == ["one"]
    assert "bold words" in sections[0].paragraphs[0]


def test_the_heading_path_records_the_trail() -> None:
    md = "# Top\nx\n\n## Middle\ny\n\n### Deep\nz\n\n## Sibling\nw\n"
    paths = {s.anchor: s.heading for s in parse_markdown(md)}
    assert paths["deep"] == "Top > Middle > Deep"
    assert paths["sibling"] == "Top > Sibling"


# ── Inline cleanup ────────────────────────────────────────────────────────────

def test_links_reduce_to_their_visible_text() -> None:
    assert strip_inline("see [the paper](https://arxiv.org/abs/1706.03762) now") \
        == "see the paper now"


def test_footnote_markers_are_stripped() -> None:
    assert strip_inline("a claim[^1] and another[^note]") == "a claim and another"


def test_a_code_fence_survives_verbatim() -> None:
    md = "## Code\n\n```python\nx = 1\n\ny = 2\n```\n\nafter\n"
    section = parse_markdown(md)[0]
    fence = next(p for p in section.paragraphs if "```" in p)
    assert "x = 1" in fence and "y = 2" in fence  # the blank line did not split it


def test_a_horizontal_rule_is_a_paragraph_break_not_content() -> None:
    section = parse_markdown("## S\nbefore\n\n---\n\nafter\n")[0]
    assert len(section.paragraphs) == 2
    assert not any("---" in p for p in section.paragraphs)


# ── Images are lifted out, not embedded as syntax ────────────────────────────

def test_images_are_collected_with_alt_text_and_position() -> None:
    md = ("![hero art](https://img.example/hero.png)\n\n"
          "# Title\n\nprose\n\n"
          "![a chart of loss curves](https://img.example/chart.png)\n\nmore\n")
    sections = parse_markdown(md)
    refs = [ref for s in sections for ref in s.images]
    assert [r.url for r in refs] == ["https://img.example/hero.png",
                                     "https://img.example/chart.png"]
    assert [r.position for r in refs] == [0, 1]
    assert refs[0].before_first_heading is True
    assert refs[1].before_first_heading is False
    assert refs[1].alt == "a chart of loss curves"
    assert refs[1].anchor == "title"


def test_image_syntax_does_not_leak_into_the_chunk_text() -> None:
    sections = parse_markdown("## S\n\n![alt](https://img.example/x.png)\n\nbody\n")
    joined = " ".join(p for s in sections for p in s.paragraphs)
    assert "img.example" not in joined
    assert "![" not in joined


def test_an_html_img_tag_is_collected_too() -> None:
    md = '## S\n\n<img src="https://img.example/y.png" alt="a diagram">\n\nbody\n'
    refs = [r for s in parse_markdown(md) for r in s.images]
    assert [(r.url, r.alt) for r in refs] == [("https://img.example/y.png",
                                               "a diagram")]


# ── Section-bounded chunking ─────────────────────────────────────────────────

def _para(word: str, n: int = 60) -> str:
    return " ".join([word] * n)


def test_no_chunk_ever_spans_two_sections() -> None:
    """The asymmetry with papers, asserted: a paper chunk may cross a page, a
    post chunk may not cross a section, because the anchor IS the citation."""
    md = "\n\n".join(f"## Section {i}\n\nshort body {i}." for i in range(1, 6))
    chunks = chunk_markdown(parse_markdown(md), max_chars=4000)
    assert len(chunks) == 5
    assert [c.anchor for c in chunks] == [f"section-{i}" for i in range(1, 6)]


def test_a_long_section_splits_into_chunks_sharing_one_anchor() -> None:
    md = f"## Long\n\n{_para('alpha')}\n\n{_para('beta')}\n\n{_para('gamma')}\n"
    chunks = chunk_markdown(parse_markdown(md), max_chars=400)
    assert len(chunks) > 1
    assert {c.anchor for c in chunks} == {"long"}


def test_the_heading_path_is_prepended_to_the_embedded_text() -> None:
    chunks = chunk_markdown(parse_markdown("# Top\n\n## Sub\n\nthe body text\n"))
    body = next(c for c in chunks if c.anchor == "sub")
    assert body.text.startswith("Top > Sub\n\n")
    assert "the body text" in body.text


def test_a_short_section_keeps_its_own_chunk() -> None:
    """A tiny trailing fragment merges WITHIN its section; a whole short
    section does not fold into the previous one — it has its own anchor, and
    merging would attribute its words to a different citation target."""
    md = "## First\n\n" + _para("filler") + "\n\n## Tiny\n\nJust this.\n"
    chunks = chunk_markdown(parse_markdown(md), max_chars=1400, min_chars=80)
    anchors = [c.anchor for c in chunks]
    assert "tiny" in anchors
    assert next(c for c in chunks if c.anchor == "tiny").text.endswith("Just this.")


def test_chunking_a_post_is_deterministic() -> None:
    md = "\n\n".join(f"## S{i}\n\n{_para(f'w{i}')}\n\n{_para(f'v{i}', 90)}"
                     for i in range(1, 4))
    sections = parse_markdown(md)
    first = chunk_markdown(sections, max_chars=500)
    second = chunk_markdown(parse_markdown(md), max_chars=500)
    assert [(c.idx, c.anchor, c.text) for c in first] == \
           [(c.idx, c.anchor, c.text) for c in second]
    assert [c.idx for c in first] == list(range(len(first)))


def test_every_paragraph_lands_in_some_chunk() -> None:
    md = "\n\n".join(f"## S{i}\n\nmarker{i} " + _para(f"body{i}", 20)
                     for i in range(6))
    joined = "\n".join(c.text for c in
                       chunk_markdown(parse_markdown(md), max_chars=600))
    for i in range(6):
        assert f"marker{i}" in joined


# ── Read guards ───────────────────────────────────────────────────────────────

def test_a_saved_web_page_is_rejected_with_a_readable_reason(tmp_path: Path) -> None:
    path = tmp_path / "saved.md"
    path.write_bytes(b"\n  <!DOCTYPE html>\n<html><body>the post</body></html>")
    with pytest.raises(SourceError, match="export the post as .md"):
        read_markdown(path)


def test_html_inside_the_body_is_not_rejected(tmp_path: Path) -> None:
    """Substack posts embed <div> and <figure> constantly — a 'contains HTML'
    check would reject the real thing."""
    path = tmp_path / "post.md"
    path.write_text("# Title\n\n<div class='pullquote'>quoted</div>\n\nbody\n")
    assert "quoted" in read_markdown(path)


def test_a_binary_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nope.md"
    path.write_bytes(b"# Title\n\x00\x01\x02binary")
    with pytest.raises(SourceError, match="NUL bytes"):
        read_markdown(path)


def test_invalid_utf8_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "latin1.md"
    path.write_bytes("# Café".encode("latin-1"))
    with pytest.raises(SourceError, match="UTF-8"):
        read_markdown(path)


def test_a_multibyte_char_across_a_read_boundary_still_decodes(tmp_path: Path) -> None:
    """The reason UTF-8 is validated on the whole file, not per 1 MB chunk: an
    em dash straddling the boundary is valid content, not a corrupt file."""
    path = tmp_path / "big.md"
    filler = "x" * ((1 << 20) - 1)          # the next char starts at byte 1MB-1
    path.write_text(f"# T\n\n{filler}—tail\n")
    assert "—tail" in read_markdown(path)


def test_curly_quotes_fold_to_ascii(tmp_path: Path) -> None:
    """Shared normalization with papers: a post and a paper quoting the same
    sentence must embed as the same sentence."""
    path = tmp_path / "quotes.md"
    path.write_text("# T\n\nIt’s a “quote”.\n")
    text = read_markdown(path)
    assert "It's a \"quote\"." in text


def test_an_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"
    path.write_text("   \n\n")
    with pytest.raises(SourceError, match="empty"):
        read_markdown(path)


# ── Key layout ────────────────────────────────────────────────────────────────

def test_post_key_is_user_scoped_and_hash_keyed() -> None:
    assert post_key("tenant-a", "abc123") == "posts/tenant-a/abc123.md"
