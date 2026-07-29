"""Unit tests for the paper fetch stage (src/ingest/paper.py).

Unlike the contract suites these import src directly (the tests container sets
PYTHONPATH=/app, same as test_error_envelope.py) — the guards under test are
pure logic that must be provable without a network or a bucket.
"""
from __future__ import annotations

import io
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from src.ingest import paper
from src.ingest.paper import (PaperSourceError, check_pdf_magic, paper_key,
                              _page_count, _stream_http)


# ── Magic bytes ───────────────────────────────────────────────────────────────

def test_a_real_pdf_header_passes() -> None:
    check_pdf_magic(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")


def test_an_html_error_page_is_a_source_error() -> None:
    with pytest.raises(PaperSourceError, match="not a PDF"):
        check_pdf_magic(b"<!doctype html><html><body>404</body></html>")


def test_a_header_after_leading_junk_still_passes() -> None:
    """The spec allows %PDF- within the first 1024 bytes; some generators
    prepend a BOM or garbage."""
    check_pdf_magic(b"\xef\xbb\xbf junk \n%PDF-1.4")


# ── Streaming guards (no network — urlopen is faked) ─────────────────────────

class _FakeResponse(io.BytesIO):
    """Just enough of an http.client.HTTPResponse for _stream_http."""

    def __init__(self, body: bytes, content_length: str | None = None):
        super().__init__(body)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_streaming_hashes_and_counts_a_valid_pdf(tmp_path: Path, monkeypatch) -> None:
    body = b"%PDF-1.4\n" + b"x" * 100
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout: _FakeResponse(body))
    dest = tmp_path / "out.pdf"
    digest, size = _stream_http("https://example.com/x.pdf", dest)
    assert size == len(body)
    assert dest.read_bytes() == body
    import hashlib
    assert digest == hashlib.sha256(body).hexdigest()


def test_a_non_pdf_body_is_a_source_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout: _FakeResponse(b"<html>nope</html>"))
    with pytest.raises(PaperSourceError, match="not a PDF"):
        _stream_http("https://example.com/x.pdf", tmp_path / "out.pdf")


def test_a_declared_oversize_body_fails_before_downloading(
    tmp_path: Path, monkeypatch
) -> None:
    """Content-Length above the cap fails without reading a byte, and the
    message names the limit."""
    huge = str((paper.MAX_PAPER_MB + 1) * (1 << 20))
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout: _FakeResponse(b"%PDF-", content_length=huge))
    with pytest.raises(PaperSourceError, match=f"{paper.MAX_PAPER_MB} MB"):
        _stream_http("https://example.com/x.pdf", tmp_path / "out.pdf")


def test_an_undeclared_oversize_body_is_caught_while_streaming(
    tmp_path: Path, monkeypatch
) -> None:
    """Servers lie about Content-Length; the running byte count is the guard."""
    monkeypatch.setattr(paper, "MAX_PAPER_MB", 1)  # keep the fixture small
    body = b"%PDF-1.4" + b"\0" * (2 << 20)         # 2 MB against a 1 MB cap
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout: _FakeResponse(body))
    with pytest.raises(PaperSourceError, match="1 MB"):
        _stream_http("https://example.com/x.pdf", tmp_path / "out.pdf")


def test_a_404_is_a_source_error_not_a_retryable_failure(
    tmp_path: Path, monkeypatch
) -> None:
    def _raise_404(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", _raise_404)
    with pytest.raises(PaperSourceError, match="HTTP 404"):
        _stream_http("https://example.com/x.pdf", tmp_path / "out.pdf")


def test_a_503_stays_retryable(tmp_path: Path, monkeypatch) -> None:
    def _raise_503(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 503, "Unavailable", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", _raise_503)
    with pytest.raises(urllib.error.HTTPError):
        _stream_http("https://example.com/x.pdf", tmp_path / "out.pdf")


# ── Page count (a generated PDF — no binary fixture in git) ──────────────────

def test_page_count_of_a_generated_pdf(tmp_path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {i + 1}")
    path = tmp_path / "three.pdf"
    doc.save(path)
    doc.close()
    assert _page_count(path) == 3


def test_an_unreadable_pdf_is_a_source_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 but nothing else")
    with pytest.raises(PaperSourceError, match="unreadable|no pages"):
        _page_count(path)


# ── Key layout ────────────────────────────────────────────────────────────────

def test_paper_key_is_user_scoped_and_hash_keyed() -> None:
    key = paper_key("tenant-a", "abc123")
    assert key == "papers/tenant-a/abc123.pdf"
