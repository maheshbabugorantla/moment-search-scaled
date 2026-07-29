"""Streaming an HTTP source into worker scratch, hashed and capped as it lands.

Extracted from ingest/paper.py when the post flow needed exactly the same
thing. What is worth sharing is not the loop — it is the *failure
classification*, which decides whether a worker slot is held for the length of
a retry ladder:

  * 4xx and NXDOMAIN are properties of the URI. Retrying cannot change them, so
    they raise SourceError and the flow's retry_condition_fn declines the retry.
  * timeouts, connection resets and 5xx are properties of the moment. They
    raise unchanged and get the retry policy.

Two copies of that judgement would drift, and the drift would be invisible
until a queue backed up.
"""
from __future__ import annotations

import hashlib
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .errors import SourceError

_CHUNK = 1 << 20  # 1 MB read granularity — bounded memory whatever the size
_HTTP_TIMEOUT_S = 60


def stream_to_file(uri: str, dest: Path, *, max_mb: int, what: str,
                   first_chunk_check: Callable[[bytes], None] | None = None,
                   ) -> tuple[str, int]:
    """GET the URI to `dest`, hashing and cap-checking as chunks arrive.

    Returns (sha256, byte_size). Never holds more than one chunk in memory.
    `what` names the media in error messages ("PDF", "markdown file").
    `first_chunk_check` sees the first chunk only, for cheap prefix guards like
    a magic-byte test — anything needing the whole file belongs in the caller.
    """
    max_bytes = max_mb * (1 << 20)

    def _check_size(byte_count: int) -> None:
        if byte_count > max_bytes:
            raise SourceError(
                f"{what} exceeds the {max_mb} MB limit "
                f"({byte_count / (1 << 20):.0f} MB so far)")

    req = urllib.request.Request(uri, headers={"User-Agent": "momentsearch/1.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        if 400 <= exc.code < 500:  # deterministic: the URI is wrong, not the network
            raise SourceError(f"HTTP {exc.code} fetching the {what}") from exc
        raise  # 5xx — the server may recover; let the retry policy have it
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, socket.gaierror):  # NXDOMAIN — no such host
            raise SourceError(f"no such host: {exc.reason}") from exc
        raise  # timeout / connection refused — retryable

    with resp:
        declared = resp.headers.get("Content-Length")
        if declared and declared.isdigit():
            _check_size(int(declared))
        h = hashlib.sha256()
        size = 0
        first = True
        with dest.open("wb") as fh:
            while chunk := resp.read(_CHUNK):
                if first:
                    if first_chunk_check is not None:
                        first_chunk_check(chunk)
                    first = False
                size += len(chunk)
                _check_size(size)  # servers lie about Content-Length; count anyway
                h.update(chunk)
                fh.write(chunk)
        if first:  # zero bytes arrived
            raise SourceError("the URI returned an empty body")
    return h.hexdigest(), size
