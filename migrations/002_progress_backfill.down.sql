-- Rollback of 002 — forget the backfill.
--
-- Backfilled pct is indistinguishable from worker-written pct, so this zeroes
-- every indexed row's pct wholesale. Acceptable: pct is derived telemetry, and
-- rolling forward again (002.up or the worker itself) restores it.
UPDATE ms_videos SET pct = 0 WHERE status = 'indexed';
