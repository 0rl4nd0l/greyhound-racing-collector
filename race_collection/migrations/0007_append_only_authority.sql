CREATE TRIGGER expected_races_append_only_update
BEFORE UPDATE ON expected_races
BEGIN SELECT RAISE(ABORT, 'expected_races is append-only'); END;

CREATE TRIGGER expected_races_append_only_delete
BEFORE DELETE ON expected_races
BEGIN SELECT RAISE(ABORT, 'expected_races is append-only'); END;

CREATE TRIGGER programme_race_observations_append_only_update
BEFORE UPDATE ON programme_race_observations
BEGIN SELECT RAISE(ABORT, 'programme_race_observations is append-only'); END;

CREATE TRIGGER programme_race_observations_append_only_delete
BEFORE DELETE ON programme_race_observations
BEGIN SELECT RAISE(ABORT, 'programme_race_observations is append-only'); END;

CREATE TRIGGER odds_attempts_append_only_update
BEFORE UPDATE ON odds_attempts
BEGIN SELECT RAISE(ABORT, 'odds_attempts is append-only'); END;

CREATE TRIGGER odds_attempts_append_only_delete
BEFORE DELETE ON odds_attempts
BEGIN SELECT RAISE(ABORT, 'odds_attempts is append-only'); END;

CREATE TRIGGER sealed_evidence_append_only_update
BEFORE UPDATE ON sealed_evidence
BEGIN SELECT RAISE(ABORT, 'sealed_evidence is append-only'); END;

CREATE TRIGGER sealed_evidence_append_only_delete
BEFORE DELETE ON sealed_evidence
BEGIN SELECT RAISE(ABORT, 'sealed_evidence is append-only'); END;

CREATE TRIGGER collection_quarantines_append_only_update
BEFORE UPDATE ON collection_quarantines
BEGIN SELECT RAISE(ABORT, 'collection_quarantines is append-only'); END;

CREATE TRIGGER collection_quarantines_append_only_delete
BEFORE DELETE ON collection_quarantines
BEGIN SELECT RAISE(ABORT, 'collection_quarantines is append-only'); END;
