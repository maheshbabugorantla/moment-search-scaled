"""Connectivity check for every managed dependency (REC-299 verify step).

Run inside the app image so it uses the same libs and the same config module.
The image only COPYs src/ and ui/, so mount this file in:

    docker compose run --rm --no-deps -e PYTHONPATH=/app \
        -v "$PWD/scripts:/app/scripts" api python /app/scripts/check_env.py

Prints PASS/FAIL per dependency. Never prints a secret — only masked hints.
Exit 0 only if every required dependency is reachable.
"""
from __future__ import annotations

import json
import sys
import traceback
import urllib.error
import urllib.request

results: list[tuple[str, bool, str]] = []


def check(name: str, required: bool = True):
    def deco(fn):
        try:
            detail = fn()
            results.append((name, True, detail or "ok"))
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if not required:
                results.append((name, True, f"skipped — {msg}"))
            else:
                results.append((name, False, msg))
                if "-v" in sys.argv:
                    traceback.print_exc()
        return fn
    return deco


def mask(value: str, keep: int = 4) -> str:
    if not value:
        return "<empty>"
    return f"…{value[-keep:]} ({len(value)} chars)"


from src import config  # noqa: E402


@check("config loaded")
def _config():
    missing = [n for n in ("DATABASE_URL", "QDRANT_URL", "LLM_API_KEY") if not getattr(config, n)]
    if missing:
        raise RuntimeError(f"unset in .env: {', '.join(missing)}")
    return (f"ADMIN_TOKEN {mask(config.ADMIN_TOKEN)} · storage={config.STORAGE_PROVIDER} "
            f"· llm={config.LLM_PROVIDER}/{config.LLM_MODEL}")


@check("Postgres (Neon)")
def _postgres():
    from src import db
    with db.pool().connection() as conn:
        ver = conn.execute("SELECT version()").fetchone()["version"]
        db.init_schema()
        n = conn.execute("SELECT count(*) AS n FROM ms_videos").fetchone()["n"]
    host = config.DATABASE_URL.split("@")[-1].split("/")[0] if "@" in config.DATABASE_URL else "?"
    return f"{ver.split(',')[0]} · host={host} · schema ok · ms_videos rows={n}"


@check("Qdrant Cloud")
def _qdrant():
    from src.rag import vector_store
    c = vector_store.client()
    existing = [x.name for x in c.get_collections().collections]
    vector_store.ensure_collection()
    if config.ENABLE_TRANSCRIPT:
        vector_store.ensure_text_collection()
    after = {x.name: c.count(x.name).count for x in c.get_collections().collections}
    return (f"reachable · collections before={existing or '[]'} · "
            f"after ensure={json.dumps(after)}")


@check("Prefect Cloud")
def _prefect():
    import os
    url = os.getenv("PREFECT_API_URL", "").rstrip("/")
    key = os.getenv("PREFECT_API_KEY", "")
    if not url or not key:
        raise RuntimeError("PREFECT_API_URL / PREFECT_API_KEY unset")
    if "/accounts/" not in url or "/workspaces/" not in url:
        raise RuntimeError(f"URL is not a Cloud workspace URL: {url[:60]}…")
    req = urllib.request.Request(
        f"{url}/deployments/filter", data=b"{}", method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        deployments = json.loads(r.read())
    names = [d.get("name") for d in deployments]
    ws = url.split("/workspaces/")[-1][:8]
    return (f"authenticated · workspace={ws}… · key {mask(key)} · "
            f"deployments={names or '[] (worker not started yet — expected)'}")


@check("OpenAI (vision)")
def _openai():
    from src import llm
    cfg = llm.env_config()
    if cfg is None:
        raise RuntimeError("no LLM configured")
    reply = llm.ping(cfg)          # sends one tiny image: proves key AND vision
    return f"{cfg.provider}/{cfg.model} answered: {reply.strip()[:60]!r}"


@check("object storage (local)")
def _storage():
    from src import storage
    key = "frames/_selfcheck/probe.txt"
    storage.put_bytes(key, b"ok", "text/plain")
    got = storage.get_bytes(key)
    storage.delete_key(key)
    return f"provider={config.STORAGE_PROVIDER} · put/get/delete ok ({got!r})"


print()
width = max(len(n) for n, _, _ in results)
for name, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
failed = [n for n, ok, _ in results if not ok]
print()
print(f"  {len(results) - len(failed)}/{len(results)} checks passed"
      + (f" — FAILED: {', '.join(failed)}" if failed else " — all dependencies reachable"))
sys.exit(1 if failed else 0)
