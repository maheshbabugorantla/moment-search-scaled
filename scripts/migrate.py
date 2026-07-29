"""Apply or roll back manifest migrations explicitly.

`up` is what db.init_schema() already does on every boot — this is here so the
pair can be round-tripped and verified, and so a migration can be applied
without restarting the stack.

`down` is never invoked automatically. It drops the columns 001 added; it does
not drop `title`, which predates the migration.

Roll back with the stack STOPPED. `init_schema()` applies every `*.up.sql` at
boot and three services call it (app, worker, seeding), so any container that
starts after a `down` immediately re-applies it. That is correct behaviour for
a stack that migrates itself on boot — it just means a rollback observed while
the stack is up will appear not to have happened.

    docker compose run --rm --no-deps -e PYTHONPATH=/app \
        -v "$PWD/scripts:/app/scripts" api python /app/scripts/migrate.py up
    ... migrate.py down
    ... migrate.py status
"""
from __future__ import annotations

import sys
from pathlib import Path

from src import db

MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"

# Columns 001 introduces — what `status` reports on.
_001_COLUMNS = ("kind", "uri", "stage", "pct")


def _apply(suffix: str) -> int:
    files = sorted(MIGRATIONS.glob(f"*.{suffix}.sql"))
    if suffix == "down":
        files.reverse()  # roll back newest first
    if not files:
        print(f"No *.{suffix}.sql in {MIGRATIONS}", file=sys.stderr)
        return 1
    with db.pool().connection() as conn:
        for path in files:
            conn.execute(path.read_text())
            print(f"applied {path.name}")
    return 0


def _status() -> int:
    with db.pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT column_name FROM information_schema.columns
             WHERE table_name = 'ms_videos' AND column_name = ANY(%s)
            """,
            (list(_001_COLUMNS),),
        ).fetchall()
        present = {r["column_name"] for r in rows}
        for col in _001_COLUMNS:
            print(f"  {col:6} {'present' if col in present else 'ABSENT'}")
        if present == set(_001_COLUMNS):
            counts = conn.execute(
                "SELECT kind, count(*) AS n FROM ms_videos GROUP BY kind ORDER BY kind"
            ).fetchall()
            print("\nrows by kind:")
            for c in counts:
                print(f"  {c['kind']:6} {c['n']}")
            missing = conn.execute(
                "SELECT count(*) AS n FROM ms_videos WHERE uri IS NULL"
            ).fetchone()["n"]
            print(f"\nrows with a NULL uri: {missing}")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "up":
        return _apply("up")
    if cmd == "down":
        return _apply("down")
    if cmd == "status":
        return _status()
    print(f"usage: migrate.py [up|down|status]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
