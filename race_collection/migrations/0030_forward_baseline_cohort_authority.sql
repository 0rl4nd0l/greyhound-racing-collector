CREATE TABLE forward_baseline_cohorts (
 cohort_id TEXT PRIMARY KEY CHECK(length(trim(cohort_id)) > 0),
 artifact_checksum TEXT NOT NULL UNIQUE CHECK(
  length(artifact_checksum)=71
  AND substr(artifact_checksum,1,7)='sha256:'
  AND substr(artifact_checksum,8) NOT GLOB '*[^0-9a-f]*'),
 frozen_at TEXT NOT NULL,
 race_count INTEGER NOT NULL CHECK(race_count=20),
 registered_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE forward_baseline_cohort_members (
 cohort_id TEXT NOT NULL REFERENCES forward_baseline_cohorts(cohort_id),
 member_ordinal INTEGER NOT NULL CHECK(member_ordinal BETWEEN 0 AND 19),
 race_id TEXT NOT NULL UNIQUE REFERENCES races(race_id),
 source_native_race_id TEXT NOT NULL UNIQUE CHECK(
  length(source_native_race_id)>0
  AND source_native_race_id NOT GLOB '*[^0-9]*'),
 member_json TEXT NOT NULL CHECK(
  json_valid(member_json) AND json_type(member_json)='object'),
 PRIMARY KEY(cohort_id,member_ordinal),
 UNIQUE(cohort_id,race_id)
);

CREATE TRIGGER forward_baseline_cohorts_append_only_update
BEFORE UPDATE ON forward_baseline_cohorts
BEGIN SELECT RAISE(ABORT,'forward baseline cohort is append-only'); END;
CREATE TRIGGER forward_baseline_cohorts_append_only_delete
BEFORE DELETE ON forward_baseline_cohorts
BEGIN SELECT RAISE(ABORT,'forward baseline cohort is append-only'); END;
CREATE TRIGGER forward_baseline_cohort_members_append_only_update
BEFORE UPDATE ON forward_baseline_cohort_members
BEGIN SELECT RAISE(ABORT,'forward baseline cohort member is append-only'); END;
CREATE TRIGGER forward_baseline_cohort_members_append_only_delete
BEFORE DELETE ON forward_baseline_cohort_members
BEGIN SELECT RAISE(ABORT,'forward baseline cohort member is append-only'); END;
