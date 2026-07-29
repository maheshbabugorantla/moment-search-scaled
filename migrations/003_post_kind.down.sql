-- 003 down — narrow the kind CHECK back to video | paper | deck.
--
-- Refuses to run while post rows exist. A down migration that leaves rows
-- violating the constraint it just installed is worse than no down migration at
-- all: the constraint would be NOT VALID in practice, and the next INSERT of an
-- unrelated kind would be the thing that finally surfaced it. Delete the post
-- sources first, deliberately, then run this.

DO $$
DECLARE
    n INT;
BEGIN
    SELECT count(*) INTO n FROM ms_videos WHERE kind = 'post';
    IF n > 0 THEN
        RAISE EXCEPTION
            'refusing to narrow ms_videos_kind_chk: % post row(s) still exist '
            '(delete them first, then re-run this migration)', n;
    END IF;

    ALTER TABLE ms_videos DROP CONSTRAINT IF EXISTS ms_videos_kind_chk;
    ALTER TABLE ms_videos ADD CONSTRAINT ms_videos_kind_chk
        CHECK (kind IN ('video', 'paper', 'deck'));
END $$;
