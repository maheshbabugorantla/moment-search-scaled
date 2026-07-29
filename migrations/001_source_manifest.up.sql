-- 001 — teach the manifest to describe a paper or a deck.
--
-- Additive only. Not one existing video row changes meaning, and no column the
-- video path reads is touched. Applied automatically by db.init_schema(), which
-- runs on every `compose up` from three callers (app, worker, seeding) — so
-- every statement here must be safe to run repeatedly.

-- Nullable first: ADD COLUMN ... NOT NULL without a default fails on a table
-- that already holds rows. Backfill, then tighten.
ALTER TABLE ms_videos ADD COLUMN IF NOT EXISTS kind  TEXT;
ALTER TABLE ms_videos ADD COLUMN IF NOT EXISTS uri   TEXT;
ALTER TABLE ms_videos ADD COLUMN IF NOT EXISTS stage TEXT;
ALTER TABLE ms_videos ADD COLUMN IF NOT EXISTS pct   INT NOT NULL DEFAULT 0;

-- Every row that predates this migration is a video.
UPDATE ms_videos SET kind = 'video' WHERE kind IS NULL;

-- `uri` is the unified source location for every kind. For videos it is
-- *derived*, not authoritative: `url` stays the column the deeplink builder and
-- the transcript fetcher read (src/rag/search.py, src/ingest/pipeline.py). See
-- the note in db.py for why the two are not collapsed.
UPDATE ms_videos
   SET uri = COALESCE(url, 'storage://' || storage_key)
 WHERE uri IS NULL;

ALTER TABLE ms_videos ALTER COLUMN kind SET NOT NULL;
ALTER TABLE ms_videos ALTER COLUMN kind SET DEFAULT 'video';

-- ADD CONSTRAINT has no IF NOT EXISTS in Postgres, so guard it explicitly —
-- otherwise the second init_schema() of the day errors the stack out.
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ms_videos_kind_chk') THEN
        ALTER TABLE ms_videos ADD CONSTRAINT ms_videos_kind_chk
            CHECK (kind IN ('video', 'paper', 'deck'));
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ms_videos_pct_chk') THEN
        ALTER TABLE ms_videos ADD CONSTRAINT ms_videos_pct_chk
            CHECK (pct BETWEEN 0 AND 100);
    END IF;
END $$;

-- The unified listing in 1.4 filters by kind within a tenant.
CREATE INDEX IF NOT EXISTS ms_videos_kind_idx ON ms_videos (user_id, kind, created_at DESC);
