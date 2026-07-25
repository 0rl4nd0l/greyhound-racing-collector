DROP TRIGGER odds_attempts_append_only_update;

ALTER TABLE odds_attempts
ADD COLUMN scheduled_due_at TEXT NOT NULL DEFAULT '';

ALTER TABLE odds_attempts
ADD COLUMN timing_policy TEXT NOT NULL DEFAULT 'adaptive-odds-timing-v1';

UPDATE odds_attempts
SET scheduled_due_at=attempted_at,
    timing_policy='adaptive-odds-timing-v1';

CREATE TRIGGER odds_attempts_append_only_update
BEFORE UPDATE ON odds_attempts
BEGIN SELECT RAISE(ABORT,'odds_attempts is append-only'); END;

CREATE TRIGGER odds_attempts_timing_exact
BEFORE INSERT ON odds_attempts WHEN
 NEW.scheduled_due_at=''
 OR NEW.timing_policy<>'adaptive-odds-timing-v1'
BEGIN SELECT RAISE(ABORT,'odds attempt timing authority is invalid'); END;

CREATE INDEX odds_attempts_by_race_due
ON odds_attempts(race_id,scheduled_due_at,attempted_at);
