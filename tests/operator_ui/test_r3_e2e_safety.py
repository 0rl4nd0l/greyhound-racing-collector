"""GHU-035C3 synthetic proof across the real API/store/worker seams.

The focused invocation includes the named lower-level collector, index, worker,
store, and verifier fixtures below. Explicit bindings prevent this suite from
manufacturing classifications which the product never emits.
"""
from __future__ import annotations

import ast
import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask
from werkzeug.security import generate_password_hash

from race_collection.synchronous_manual_capture import VerifiedCurrentRaceIndex
from src.operator_ui.job_store import JobInput, JobStore, Phase
from src.operator_ui.prediction_worker import ServerChoice, WorkerConfig, run_once
from src.operator_ui.r3_api import R3Services, ResolvedSubmission, install_r3_api
from src.operator_ui.security import install_connected_mode
from tests.operator_ui.test_prediction_worker import sealed_blocked

NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)
DIGEST = hashlib.sha256(b"ghu-035c1").hexdigest()
RACE = {"race_number": 5, "venue": "RICH", "race_date": "2026-08-01", "url": "https://www.thedogs.com.au/racing/richmond/2026-08-01/5"}
RACE_ID = "race-20260801-richmond-r05"
ROOT = Path(__file__).resolve().parents[2]

# Complete source modules containing these genuine fixtures are part of the
# recorded focused command; this map makes fixture drift fail visibly.
BOUND_FIXTURES = {
    "tests/race_collection/test_synchronous_manual_capture.py": {
        "test_current_race_index_publication_is_atomic_bounded_and_source_sealed",
        "test_current_race_index_rejects_stale_or_changed_source",
        "test_current_index_rejects_noncanonical_aliases",
        "test_safe_file_bytes_rejects_unsafe_types_and_sizes",
        "test_v2_requires_matching_successful_retained_publication",
    },
    "tests/race_collection/test_manual_prediction_collector_request.py": {
        "test_snapshot_counts_non_json_entries_before_filtering",
        "test_snapshot_rejects_oversized_member",
        "test_snapshot_rejects_intermediate_symlink",
        "test_snapshot_rejects_each_retained_member_changing_after_read",
        "test_request_claim_ready_response_and_consume_once",
        "test_duplicate_request_claim_and_response_fail_closed",
    },
    "tests/operator_ui/test_live_adapters.py": {
        "test_real_collector_rejection_codes_map_truthfully",
        "test_prediction_verified_records_preserve_terminal_distinctions",
        "test_no_browser_path_shell_scan_write_service_or_database_surface",
    },
    "tests/operator_ui/test_job_store.py": {
        "test_idempotency_duplicate_conflict_cross_actor_and_raw_key_absent",
        "test_only_verifier_capability_can_finalize_producer_completion",
        "test_every_sealed_producer_blocker_preserves_exact_identity_through_verifier_and_reopen",
    },
    "tests/operator_ui/test_prediction_worker.py": {
        "test_exact_argv_and_forbidden_surface",
        "test_valid_launch_is_nonterminal_producer_completion_and_restart_never_launches",
        "test_timeout_reap_truth_is_durable_and_never_retries",
        "test_output_caps_classify_without_deadlock",
    },
    "tests/operator_ui/test_r3_api.py": {
        "test_level2_csrf_exact_schema_idempotency_poll_and_actor_isolation",
        "test_level1_cannot_submit_or_read_and_resolution_blockers_disclose_no_job",
    },
}


def test_bound_genuine_fixtures_remain_present():
    for relative, expected in BOUND_FIXTURES.items():
        tree = ast.parse((ROOT / relative).read_bytes(), filename=relative)
        present = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert expected <= present, (relative, sorted(expected - present))


class Process:
    def __init__(self, stdout: bytes):
        self.pid = 3501
        self.stdout, self.stderr = io.BytesIO(stdout), io.BytesIO()
        self.returncode = 2

    def wait(self, timeout): return self.returncode
    def terminate(self): raise AssertionError("fixture must not be reaped")
    def poll(self): return self.returncode


def test_one_authenticated_submission_reaches_real_worker_once_without_false_ready(tmp_path):
    """One POST crosses API/store/worker; only the verifier may make it ready."""
    files = []
    for name in ("config.json", "model.json", "manifest.json", "schema.json"):
        path = tmp_path / name
        path.write_bytes(b"ghu-035c1")
        files.append(path)
    choice = ServerChoice(files[0], "manual-default", DIGEST, "market_form_residual_v1", DIGEST, DIGEST, DIGEST, *files[1:])
    config = WorkerConfig(Path("/usr/bin/python3"), ROOT, {"latest-research": choice}, tmp_path / "canonical.db", tmp_path / "output", (tmp_path / "evidence-a",), tmp_path / "requests", tmp_path / "index.json", tmp_path / "evidence", 1, 45.0, 90.0, 2)

    app = Flask(__name__)
    app.config.update(TESTING=True, OPERATOR_UI_CONNECTED_MODE=True, OPERATOR_UI_SECRET_KEY="s" * 48,
        OPERATOR_UI_USERNAME="operator", OPERATOR_UI_PASSWORD_HASH=generate_password_hash("safe-password"),
        OPERATOR_UI_LEVEL=2, OPERATOR_UI_AUDIT_DB_PATH=str(tmp_path / "audit.db"),
        OPERATOR_UI_JOB_DB_PATH=str(tmp_path / "jobs.db"), DATABASE_PATH=str(tmp_path / "canonical.db"),
        OPERATOR_UI_DEPLOYED_COMMIT="c" * 40, OPERATOR_UI_DEPLOYED_TREE="d" * 40,
        OPERATOR_UI_DEPLOYED_VERSION="fixture", OPERATOR_UI_CLOCK=lambda: NOW)
    install_connected_mode(app)
    store = JobStore(tmp_path / "jobs.db", separate_from=(tmp_path / "audit.db", tmp_path / "canonical.db"))
    runners = ({"box": 1, "name": "ALPHA", "identity": "ALPHA"},)
    inp = JobInput(RACE_ID, "2026-08-01T01:00:00+00:00", DIGEST, "latest-research", "market_form_residual_v1", DIGEST, DIGEST, DIGEST, "manual-default", DIGEST, "auto", runners)
    resolved = lambda selected, now: ResolvedSubmission(inp, runners)
    launches = []

    def index(**kwargs):
        return VerifiedCurrentRaceIndex("collector_current_race_index_v2", "run", "2026-08-01T00:00:00Z", DIGEST, b"{}", ({"race_id": RACE_ID, "jump_datetime": "2026-08-01T01:00:00+00:00", "runner_set_sha256": DIGEST},), "source.json", DIGEST, DIGEST, DIGEST, DIGEST)

    def launch(job_id, confirm):
        launches.append(job_id)
        output = sealed_blocked(store.get(job_id), "POST_JUMP")
        run_once(store, job_id, config, now=lambda: NOW, confirm_audit=confirm,
                 popen=lambda *args, **kwargs: Process(output),
                 reader=index)

    install_r3_api(app, R3Services(store, resolved, launch, lambda job: None, clock=lambda: NOW, rate_limit=20))
    client = app.test_client()
    token = client.get("/operator-ui/login", base_url="https://localhost").get_json()["csrf_token"]
    token = client.post("/operator-ui/login", base_url="https://localhost", data={"username": "operator", "password": "safe-password", "csrf_token": token}).get_json()["csrf_token"]
    payload = {"race_id": RACE_ID, "model_id": "latest-research", "config_id": "manual-default", "odds_source_id": "auto", "idempotency_key": "12345678-1234-4123-8123-123456789abc"}
    first = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=payload, headers={"X-CSRF-Token": token})
    duplicate = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=payload, headers={"X-CSRF-Token": token})
    assert first.status_code == 202 and duplicate.status_code == 200
    assert duplicate.get_json()["job_id"] == first.get_json()["job_id"]
    assert launches == [first.get_json()["job_id"]]
    job = store.get(first.get_json()["job_id"])
    assert job.phase is Phase.PRODUCER_COMPLETED
    assert job.reason == "PRODUCER_PREDICTION_BLOCKED:POST_JUMP"
    assert all(event["phase"] != Phase.PREDICTION_READY.value for event in store.events(job.job_id))
    assert store.verify()
