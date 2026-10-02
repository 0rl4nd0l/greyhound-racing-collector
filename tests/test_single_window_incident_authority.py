"""One prospectively issued window under the exact renewed October2 authority."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pytest
from race_collection.incident_engineering import load_incident_authority, validate_incident_lease, incident_usage
from tests.fixtures.incident_engineering_case import make_incident
from tests.test_incident_engineering import october2_authority, put

# Exact retained user scope; no provider or outcome data.
AMENDMENT_BYTES = b'{\n  "schema": "live_first_recovery_authority_amendment_v1",\n  "issued_at": "2026-10-02T04:41:50.536547+00:00",\n  "authority_reference": "user:20261002-live-first-resume-03",\n  "user_direction": "Resume using actual live races as the primary testing surface; more short failed attempts are authorized; previous amount-of-attempts and fixed debugging-window limits need not bind future work; notify when live collection begins.",\n  "superseded_restrictions": [\n    "October2 overall two-acceptance-attempt maximum",\n    "Fixed duration of every debugging attempt",\n    "Repeated complete synthetic90minute lifecycle as a prerequisite after a narrow reviewed correction"\n  ],\n  "preserved_requirements": [\n    "Every previous consumed attempt/counter/denial remains immutable",\n    "Per-run finite allowances, measured and separate from provider permission",\n    "Existing per-minute source controls and denial handling",\n    "Root sole live provider/installation/launch owner",\n    "No scientific or weekend allocation borrowing; no canary clearing or schedule amendment",\n    "Private official-result processing and strict identity/completeness",\n    "Frozen models, no retraining, betting or performance evaluation",\n    "Source/configuration frozen within each measured acceptance run",\n    "One genuine uninterrupted90minute final-release acceptance remains required",\n    "Stop new collection21:00AEST; collection cleanup21:30AEST; resultsdeadlineOct4noonMelbourne"\n  ],\n  "first_failed_window": {\n    "path": "/home/l4nd0/greyhound-recovery-20261002/root/window-001-failed-restored.json",\n    "sha256": "62ce74d1123c9b9f258ce3f84ef6f472d919a4dd9b0a50458e6b3c4a4982c011"\n  },\n  "prior_unconsumed_second_configuration": {\n    "path": "/home/l4nd0/greyhound-recovery-20261002/root/prepared-window-002-corrected.json",\n    "sha256": "c523da857e9fd9f9c10c3a2a15db1dd96f696d83e8080ef484da6b9392353182"\n  },\n  "prior_second_disposition": "PRESERVED_UNARMED_NOT_TO_BE_LAUNCHED",\n  "source_commit": "da798d0c0feda22acb18414985972bd8454f552f",\n  "planned_next_attempt": "15:00\\u201316:30AEST native90minute engineering run; if it fails, preserve failure and correct from live evidence before a subsequent fresh finite authority",\n  "local_request_caps_are_provider_permission": false\n}'

@pytest.fixture
def single(tmp_path):
    value, ref = october2_authority(make_incident(tmp_path))
    amendment = tmp_path / "live-first-amendment.json"
    amendment.write_bytes(AMENDMENT_BYTES)
    value.update(schema_version="collector_incident_engineering_authority_20261002_v2",
                 issued_at="2026-10-02T17:30:00+10:00",
                 live_first_amendment={"path": str(amendment), "sha256": hashlib.sha256(AMENDMENT_BYTES).hexdigest()},
                 slots=[{"id":"001", "starts_at":"2026-10-02T18:00:00+10:00",
                         "ends_at":"2026-10-02T19:30:00+10:00", "cleanup_by":"2026-10-02T20:00:00+10:00"}])
    return value, put(Path(ref["path"]), value)

def test_single_window_valid_without_a_fictitious_second_slot(single):
    value, ref = single
    assert load_incident_authority(ref) == value

@pytest.mark.parametrize("defect", ["missing_amendment", "changed_amendment", "rehashed_weakened_amendment", "issued_before_amendment", "already_started", "two_slots", "wrong_slot", "short_window", "late_collection", "late_cleanup", "late_results", "python_cap", "browser_cap", "capture_cap", "source_cap", "result_cap", "study_enrolment"])
def test_single_window_cannot_expand_or_revive_authority(single, defect):
    value, ref = single
    slot = value["slots"][0]
    if defect == "missing_amendment": del value["live_first_amendment"]
    elif defect in {"changed_amendment", "rehashed_weakened_amendment"}:
        p = Path(value["live_first_amendment"]["path"])
        amendment = json.loads(p.read_bytes()); amendment["local_request_caps_are_provider_permission"] = True
        updated = put(p, amendment)
        if defect == "rehashed_weakened_amendment": value["live_first_amendment"] = updated
    elif defect == "issued_before_amendment": value["issued_at"] = "2026-10-02T14:00:00+10:00"
    elif defect == "already_started": value["issued_at"] = slot["starts_at"]
    elif defect == "two_slots": value["slots"].append(deepcopy(slot))
    elif defect == "wrong_slot": slot["id"] = "002"
    elif defect == "short_window": slot["ends_at"] = "2026-10-02T19:00:00+10:00"
    elif defect == "late_collection": slot.update(starts_at="2026-10-02T19:31:00+10:00", ends_at="2026-10-02T21:01:00+10:00", cleanup_by="2026-10-02T21:30:00+10:00")
    elif defect == "late_cleanup": slot["cleanup_by"] = "2026-10-02T21:31:00+10:00"
    elif defect == "late_results": value["result_deadline"] = "2026-10-04T12:01:00+11:00"
    elif defect == "python_cap": value["max_python_requests_per_window"] += 1
    elif defect == "browser_cap": value["max_browser_navigations_per_window"] += 1
    elif defect == "capture_cap": value["max_capture_attempts_per_window"] += 1
    elif defect == "source_cap": value["max_source_operations_per_window"] += 1
    elif defect == "result_cap": value["max_result_requests_per_window"] += 1
    else: value["study_enrolment"] = True
    with pytest.raises(ValueError, match="invalid_incident_authority"):
        load_incident_authority(put(Path(ref["path"]), value))

def test_last_single_window_can_finish_at_hard_day_cutoffs(single):
    value, ref = single
    value["slots"][0].update(starts_at="2026-10-02T19:30:00+10:00", ends_at="2026-10-02T21:00:00+10:00", cleanup_by="2026-10-02T21:30:00+10:00")
    assert load_incident_authority(put(Path(ref["path"]), value))["slots"][0]["ends_at"] == value["collection_stop_at"]

def test_single_lease_cannot_outlive_collection_or_result_deadline(single):
    value, ref = single
    start = datetime.fromisoformat(value["slots"][0]["starts_at"]).timestamp()
    row = dict(incident_authority=ref, incident_authority_sha256=ref["sha256"], incident_id=value["incident_id"], incident_slot="001", incident_kind="prediction", reference=value["authority_reference"]+":slot:001", prior_phase="OPEN", authorized_at=start-60, expires_at=start+5400, max_operations=192)
    assert validate_incident_lease(row) == value
    row["expires_at"] += 1
    with pytest.raises(ValueError, match="invalid_incident_source_lease"): validate_incident_lease(row)
    row.update(incident_kind="results", reference=value["authority_reference"]+":slot:001:results", max_operations=96, expires_at=datetime.fromisoformat(value["result_deadline"]).timestamp()+1)
    with pytest.raises(ValueError, match="invalid_incident_source_lease"): validate_incident_lease(row)

def test_historical_v1_authorities_and_consumed_counters_remain_separate(single,tmp_path):
    new, newref=single
    oct1, ref1=make_incident(tmp_path/"oct1")
    oct2, ref2=october2_authority(make_incident(tmp_path/"oct2"))
    ledger={"incident_request_usage":{}}
    for v,r in [(oct1,ref1),(oct2,ref2),(new,newref)]:
        assert load_incident_authority(r)==v
        ledger["incident_request_usage"][r["sha256"]+":001"]={"incident_authority":r,"incident_authority_sha256":r["sha256"],"incident_id":v["incident_id"],"incident_slot":"001","counts":{"prediction":3,"results":1}}
    before=deepcopy(ledger)
    assert incident_usage(ledger,"SYNTHETIC")["logical_requests"]==12
    assert incident_usage(ledger,"SYNTHETIC",authority_sha256=newref["sha256"])["logical_requests"]==4
    assert ledger==before
    for v,r in [(oct1,ref1),(oct2,ref2)]:
        v["slots"]=v["slots"][:1]
        with pytest.raises(ValueError,match="invalid_incident_authority"):load_incident_authority(put(Path(r["path"]),v))
