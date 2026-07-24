CREATE TRIGGER run_observations_artifact_checksum_insert
BEFORE INSERT ON run_observations WHEN length(NEW.artifact_checksum) <> 71 OR substr(NEW.artifact_checksum, 1, 7) <> 'sha256:' OR substr(NEW.artifact_checksum, 8) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT, 'invalid run_observations.artifact_checksum'); END;
CREATE TRIGGER run_observations_artifact_checksum_update
BEFORE UPDATE OF artifact_checksum ON run_observations WHEN length(NEW.artifact_checksum) <> 71 OR substr(NEW.artifact_checksum, 1, 7) <> 'sha256:' OR substr(NEW.artifact_checksum, 8) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT, 'invalid run_observations.artifact_checksum'); END;

CREATE TRIGGER expected_races_programme_checksum_insert
BEFORE INSERT ON expected_races WHEN length(NEW.programme_checksum) <> 71 OR substr(NEW.programme_checksum, 1, 7) <> 'sha256:' OR substr(NEW.programme_checksum, 8) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT, 'invalid expected_races.programme_checksum'); END;
CREATE TRIGGER expected_races_programme_checksum_update
BEFORE UPDATE OF programme_checksum ON expected_races WHEN length(NEW.programme_checksum) <> 71 OR substr(NEW.programme_checksum, 1, 7) <> 'sha256:' OR substr(NEW.programme_checksum, 8) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT, 'invalid expected_races.programme_checksum'); END;

CREATE TRIGGER odds_attempts_artifact_checksum_insert
BEFORE INSERT ON odds_attempts WHEN NEW.artifact_checksum IS NOT NULL AND (length(NEW.artifact_checksum) <> 71 OR substr(NEW.artifact_checksum, 1, 7) <> 'sha256:' OR substr(NEW.artifact_checksum, 8) GLOB '*[^0-9a-f]*')
BEGIN SELECT RAISE(ABORT, 'invalid odds_attempts.artifact_checksum'); END;
CREATE TRIGGER odds_attempts_artifact_checksum_update
BEFORE UPDATE OF artifact_checksum ON odds_attempts WHEN NEW.artifact_checksum IS NOT NULL AND (length(NEW.artifact_checksum) <> 71 OR substr(NEW.artifact_checksum, 1, 7) <> 'sha256:' OR substr(NEW.artifact_checksum, 8) GLOB '*[^0-9a-f]*')
BEGIN SELECT RAISE(ABORT, 'invalid odds_attempts.artifact_checksum'); END;
CREATE TRIGGER odds_attempts_runner_mapping_checksum_insert
BEFORE INSERT ON odds_attempts WHEN NEW.runner_mapping_checksum IS NOT NULL AND (length(NEW.runner_mapping_checksum) <> 71 OR substr(NEW.runner_mapping_checksum, 1, 7) <> 'sha256:' OR substr(NEW.runner_mapping_checksum, 8) GLOB '*[^0-9a-f]*')
BEGIN SELECT RAISE(ABORT, 'invalid odds_attempts.runner_mapping_checksum'); END;
CREATE TRIGGER odds_attempts_runner_mapping_checksum_update
BEFORE UPDATE OF runner_mapping_checksum ON odds_attempts WHEN NEW.runner_mapping_checksum IS NOT NULL AND (length(NEW.runner_mapping_checksum) <> 71 OR substr(NEW.runner_mapping_checksum, 1, 7) <> 'sha256:' OR substr(NEW.runner_mapping_checksum, 8) GLOB '*[^0-9a-f]*')
BEGIN SELECT RAISE(ABORT, 'invalid odds_attempts.runner_mapping_checksum'); END;

CREATE TRIGGER field_evidence_artifact_checksum_insert
BEFORE INSERT ON field_evidence WHEN length(NEW.artifact_checksum) <> 71 OR substr(NEW.artifact_checksum, 1, 7) <> 'sha256:' OR substr(NEW.artifact_checksum, 8) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT, 'invalid field_evidence.artifact_checksum'); END;
CREATE TRIGGER field_evidence_artifact_checksum_update
BEFORE UPDATE OF artifact_checksum ON field_evidence WHEN length(NEW.artifact_checksum) <> 71 OR substr(NEW.artifact_checksum, 1, 7) <> 'sha256:' OR substr(NEW.artifact_checksum, 8) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT, 'invalid field_evidence.artifact_checksum'); END;

CREATE TRIGGER sealed_evidence_checksums_insert
BEFORE INSERT ON sealed_evidence WHEN length(NEW.raw_manifest_checksum) <> 71 OR substr(NEW.raw_manifest_checksum, 1, 7) <> 'sha256:' OR substr(NEW.raw_manifest_checksum, 8) GLOB '*[^0-9a-f]*' OR length(NEW.normalized_checksum) <> 71 OR substr(NEW.normalized_checksum, 1, 7) <> 'sha256:' OR substr(NEW.normalized_checksum, 8) GLOB '*[^0-9a-f]*' OR length(NEW.odds_checksum) <> 71 OR substr(NEW.odds_checksum, 1, 7) <> 'sha256:' OR substr(NEW.odds_checksum, 8) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT, 'invalid sealed_evidence checksum'); END;
CREATE TRIGGER sealed_evidence_checksums_update
BEFORE UPDATE OF raw_manifest_checksum, normalized_checksum, odds_checksum ON sealed_evidence WHEN length(NEW.raw_manifest_checksum) <> 71 OR substr(NEW.raw_manifest_checksum, 1, 7) <> 'sha256:' OR substr(NEW.raw_manifest_checksum, 8) GLOB '*[^0-9a-f]*' OR length(NEW.normalized_checksum) <> 71 OR substr(NEW.normalized_checksum, 1, 7) <> 'sha256:' OR substr(NEW.normalized_checksum, 8) GLOB '*[^0-9a-f]*' OR length(NEW.odds_checksum) <> 71 OR substr(NEW.odds_checksum, 1, 7) <> 'sha256:' OR substr(NEW.odds_checksum, 8) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT, 'invalid sealed_evidence checksum'); END;

CREATE TRIGGER programme_observations_checksum_insert
BEFORE INSERT ON programme_race_observations WHEN length(NEW.programme_checksum) <> 71 OR substr(NEW.programme_checksum, 1, 7) <> 'sha256:' OR substr(NEW.programme_checksum, 8) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT, 'invalid programme_race_observations.programme_checksum'); END;
CREATE TRIGGER programme_observations_checksum_update
BEFORE UPDATE OF programme_checksum ON programme_race_observations WHEN length(NEW.programme_checksum) <> 71 OR substr(NEW.programme_checksum, 1, 7) <> 'sha256:' OR substr(NEW.programme_checksum, 8) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT, 'invalid programme_race_observations.programme_checksum'); END;

UPDATE run_observations SET artifact_checksum = artifact_checksum;
UPDATE expected_races SET programme_checksum = programme_checksum;
UPDATE odds_attempts SET artifact_checksum = artifact_checksum, runner_mapping_checksum = runner_mapping_checksum;
UPDATE field_evidence SET artifact_checksum = artifact_checksum;
UPDATE sealed_evidence SET raw_manifest_checksum = raw_manifest_checksum, normalized_checksum = normalized_checksum, odds_checksum = odds_checksum;
UPDATE programme_race_observations SET programme_checksum = programme_checksum;
