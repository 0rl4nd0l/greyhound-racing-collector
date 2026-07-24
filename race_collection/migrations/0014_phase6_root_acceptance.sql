CREATE TABLE phase6_runs (
    run_id TEXT PRIMARY KEY CHECK(run_id GLOB 'op_[0-9a-f]*' AND length(run_id)=35),
    run_kind TEXT NOT NULL CHECK(run_kind IN ('evaluation','promotion','training','tuning','registration','forecast_service')),
    started_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE phase6_forecast_artifacts (
    race_id TEXT NOT NULL,
    bundle_id TEXT NOT NULL REFERENCES canonical_model_bundles(bundle_id),
    bundle_checksum TEXT NOT NULL,
    evidence_checksum TEXT NOT NULL,
    forecast_checksum TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    evaluation_run_id TEXT NOT NULL REFERENCES phase6_runs(run_id),
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    PRIMARY KEY(race_id,bundle_id),
    FOREIGN KEY(bundle_id,bundle_checksum) REFERENCES canonical_model_bundles(bundle_id,bundle_checksum)
);

CREATE TABLE phase6_policy_registry (
    policy_id TEXT PRIMARY KEY,
    policy_checksum TEXT NOT NULL UNIQUE,
    artifact_checksum TEXT NOT NULL UNIQUE,
    registered_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE phase6_trusted_evaluations (
    evidence_id TEXT PRIMARY KEY REFERENCES phase6_evaluation_evidence(evidence_id),
    evaluation_run_id TEXT NOT NULL UNIQUE REFERENCES phase6_runs(run_id),
    champion_assignment_id TEXT NOT NULL REFERENCES canonical_serving_assignments(assignment_id),
    challenger_registered_at TEXT NOT NULL,
    policy_checksum TEXT NOT NULL REFERENCES phase6_policy_registry(policy_checksum),
    report_checksum TEXT NOT NULL UNIQUE,
    sealed_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE phase6_racing_day_schedule (
    racing_day_id TEXT PRIMARY KEY REFERENCES racing_days(racing_day_id),
    predecessor_racing_day_id TEXT UNIQUE REFERENCES racing_days(racing_day_id),
    programme_checksum TEXT NOT NULL,
    scheduled_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE phase6_probation_day_auth (
    probation_id TEXT NOT NULL REFERENCES phase6_probation_states(probation_id),
    racing_day_id TEXT NOT NULL REFERENCES phase6_racing_day_schedule(racing_day_id),
    programme_checksum TEXT NOT NULL,
    PRIMARY KEY(probation_id,racing_day_id)
);

CREATE TABLE phase6_next_day_assignments (
    assignment_id TEXT PRIMARY KEY REFERENCES canonical_serving_assignments(assignment_id),
    effective_racing_day_id TEXT NOT NULL UNIQUE REFERENCES phase6_racing_day_schedule(racing_day_id),
    rollback_assignment_id TEXT NOT NULL REFERENCES canonical_serving_assignments(assignment_id),
    promotion_record_id TEXT NOT NULL UNIQUE REFERENCES phase6_promotion_records(promotion_record_id),
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE phase6_rollback_records (
    rollback_id TEXT PRIMARY KEY,
    staged_assignment_id TEXT NOT NULL UNIQUE REFERENCES phase6_next_day_assignments(assignment_id),
    restored_assignment_id TEXT NOT NULL REFERENCES canonical_serving_assignments(assignment_id),
    reason TEXT NOT NULL CHECK(length(trim(reason)) > 0),
    rolled_back_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TRIGGER phase6_runs_append_only_update BEFORE UPDATE ON phase6_runs BEGIN SELECT RAISE(ABORT,'phase6 runs are append-only'); END;
CREATE TRIGGER phase6_runs_append_only_delete BEFORE DELETE ON phase6_runs BEGIN SELECT RAISE(ABORT,'phase6 runs are append-only'); END;
CREATE TRIGGER phase6_forecast_append_only_update BEFORE UPDATE ON phase6_forecast_artifacts BEGIN SELECT RAISE(ABORT,'phase6 forecasts are append-only'); END;
CREATE TRIGGER phase6_forecast_append_only_delete BEFORE DELETE ON phase6_forecast_artifacts BEGIN SELECT RAISE(ABORT,'phase6 forecasts are append-only'); END;
CREATE TRIGGER phase6_policy_append_only_update BEFORE UPDATE ON phase6_policy_registry BEGIN SELECT RAISE(ABORT,'phase6 policies are append-only'); END;
CREATE TRIGGER phase6_policy_append_only_delete BEFORE DELETE ON phase6_policy_registry BEGIN SELECT RAISE(ABORT,'phase6 policies are append-only'); END;
CREATE TRIGGER phase6_trusted_evaluations_append_only_update BEFORE UPDATE ON phase6_trusted_evaluations BEGIN SELECT RAISE(ABORT,'trusted evaluations are append-only'); END;
CREATE TRIGGER phase6_trusted_evaluations_append_only_delete BEFORE DELETE ON phase6_trusted_evaluations BEGIN SELECT RAISE(ABORT,'trusted evaluations are append-only'); END;
CREATE TRIGGER phase6_schedule_append_only_update BEFORE UPDATE ON phase6_racing_day_schedule BEGIN SELECT RAISE(ABORT,'racing day schedule is append-only'); END;
CREATE TRIGGER phase6_schedule_append_only_delete BEFORE DELETE ON phase6_racing_day_schedule BEGIN SELECT RAISE(ABORT,'racing day schedule is append-only'); END;
CREATE TRIGGER phase6_probation_auth_append_only_update BEFORE UPDATE ON phase6_probation_day_auth BEGIN SELECT RAISE(ABORT,'probation day authentication is append-only'); END;
CREATE TRIGGER phase6_probation_auth_append_only_delete BEFORE DELETE ON phase6_probation_day_auth BEGIN SELECT RAISE(ABORT,'probation day authentication is append-only'); END;
CREATE TRIGGER phase6_next_assignments_append_only_update BEFORE UPDATE ON phase6_next_day_assignments BEGIN SELECT RAISE(ABORT,'next-day assignments are append-only'); END;
CREATE TRIGGER phase6_next_assignments_append_only_delete BEFORE DELETE ON phase6_next_day_assignments BEGIN SELECT RAISE(ABORT,'next-day assignments are append-only'); END;
CREATE TRIGGER phase6_rollbacks_append_only_update BEFORE UPDATE ON phase6_rollback_records BEGIN SELECT RAISE(ABORT,'rollbacks are append-only'); END;
CREATE TRIGGER phase6_rollbacks_append_only_delete BEFORE DELETE ON phase6_rollback_records BEGIN SELECT RAISE(ABORT,'rollbacks are append-only'); END;
