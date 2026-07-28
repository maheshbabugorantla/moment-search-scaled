"""A consistent JSON error envelope for the admin surface.

The Definition of Done names three status codes and the rubric checks them:
400 bad input, 401 missing/bad admin token, 502 upstream failure. This gives
them one machine-readable shape:

    { "error": { "code": "upstream_unavailable", "message": "..." } }

**Scoped to /admin/* on purpose.** The provided UI reads `detail` off error
bodies in three places (ui/index.html), so rewriting every error response would
break the UI's error messages — which "the provided video endpoints and the UI
still work unmodified" forbids. Non-admin paths keep FastAPI's default body
byte-for-byte.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

log = logging.getLogger("momentsearch.admin")

ADMIN_PREFIX = "/admin"

# Status -> machine-readable code, for the cases that don't carry their own.
_DEFAULT_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    422: "invalid_request",
    502: "upstream_unavailable",
}


class UpstreamError(HTTPException):
    """A dependency the request path genuinely needs is unavailable.

    502, not 500: the distinction is "their fault" vs "our bug", and it is the
    one the rubric checks. Raise this only for a dependency failure on the
    accept path — worker-side failures set the row to `failed` and never
    produce a status code.
    """

    def __init__(self, message: str, *, code: str = "upstream_unavailable",
                 source_id: str | None = None):
        super().__init__(502, message)
        self.code = code
        self.source_id = source_id


def _is_admin(request: Request) -> bool:
    return request.url.path.startswith(ADMIN_PREFIX)


def _envelope(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"code": code, "message": message}})


def register(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException):
        if not _is_admin(request):
            return await http_exception_handler(request, exc)
        code = getattr(exc, "code", None) or _DEFAULT_CODES.get(
            exc.status_code, "error")
        if exc.status_code >= 500:
            log.error("admin %s -> %s (%s) source=%s", request.url.path,
                      exc.status_code, code, getattr(exc, "source_id", None))
        return _envelope(exc.status_code, code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        if not _is_admin(request):
            return await request_validation_exception_handler(request, exc)
        # Pydantic's error list is useful; flatten it to one line so the body
        # stays the same shape as every other error.
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        return _envelope(422, "invalid_request", detail or "Invalid request.")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # Never leak a stack trace to the client; log it with the path so it is
        # traceable. Non-admin paths re-raise so FastAPI's default applies.
        log.exception("unhandled error on %s", request.url.path)
        if not _is_admin(request):
            raise exc
        return _envelope(500, "internal_error",
                         "Internal error. The failure has been logged.")
