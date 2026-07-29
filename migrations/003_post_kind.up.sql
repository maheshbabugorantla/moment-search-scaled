-- 003 — teach the manifest that a source can be a markdown post.
--
-- One CHECK constraint, widened. `post` is the third document kind: markdown in,
-- cited by heading anchor rather than page number (the anchor survives a
-- re-render and deeplinks the live page; a page number is an artifact of the
-- renderer's margins).
--
-- Re-runnable, like every migration here: db.init_schema() executes every
-- migrations/*.up.sql on every boot, from three callers (app, worker, seeding).
-- 001 creates ms_videos_kind_chk only when a constraint of that NAME is absent,
-- so it will not undo this one; _migrations() globs in sorted order, so 001
-- always runs first on a fresh database and this widens what it created.
--
-- The rewrite is guarded on the constraint's current definition rather than run
-- unconditionally: DROP + ADD re-validates the whole table, and there is no
-- reason to pay that on every `compose up` once the constraint is already right.

DO $$
DECLARE
    current_def TEXT;
BEGIN
    SELECT pg_get_constraintdef(oid) INTO current_def
      FROM pg_constraint WHERE conname = 'ms_videos_kind_chk';

    IF current_def IS NULL OR current_def NOT LIKE '%''post''%' THEN
        ALTER TABLE ms_videos DROP CONSTRAINT IF EXISTS ms_videos_kind_chk;
        ALTER TABLE ms_videos ADD CONSTRAINT ms_videos_kind_chk
            CHECK (kind IN ('video', 'paper', 'deck', 'post'));
    END IF;
END $$;
