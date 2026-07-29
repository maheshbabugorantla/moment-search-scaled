-- Rollback of 001. Run explicitly (scripts/migrate.py down) with the stack
-- STOPPED; nothing invokes this automatically. init_schema() re-applies the up
-- migration on every service boot, so a rollback taken while the stack is
-- running is undone by the next container that starts.
--
-- `title` is deliberately NOT dropped. It predates this migration — the issue
-- listed it as a new column, but the schema already had it — and dropping it
-- would destroy admin-supplied titles on every existing video.

ALTER TABLE ms_videos DROP CONSTRAINT IF EXISTS ms_videos_kind_chk;
ALTER TABLE ms_videos DROP CONSTRAINT IF EXISTS ms_videos_pct_chk;
DROP INDEX IF EXISTS ms_videos_kind_idx;

ALTER TABLE ms_videos DROP COLUMN IF EXISTS kind;
ALTER TABLE ms_videos DROP COLUMN IF EXISTS uri;
ALTER TABLE ms_videos DROP COLUMN IF EXISTS stage;
ALTER TABLE ms_videos DROP COLUMN IF EXISTS pct;
