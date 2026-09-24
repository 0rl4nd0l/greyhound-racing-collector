import json

import pytest

from race_collection.live_phase_checkpoint import PhaseCheckpoint


def checkpoint(tmp_path, identity="source"):
    return PhaseCheckpoint(
        tmp_path / "state.json",
        identity=identity,
        cycle_id="cycle",
        output_dir=tmp_path / "evidence",
    )


def test_restart_never_replays_ambiguous_started_work(tmp_path):
    original = checkpoint(tmp_path)
    original.begin("capture", {"race_id": "fixture"}, "2026-07-19T12:00:00+10:00")
    before = original.path.read_bytes()
    with pytest.raises(ValueError, match="interrupted_requires_reconciliation"):
        checkpoint(tmp_path)
    assert original.path.read_bytes() == before


def test_resume_only_completed_boundary_with_exact_identity(tmp_path):
    original = checkpoint(tmp_path)
    original.begin("refresh", {}, "2026-07-19T12:00:00+10:00")
    original.value["pending"] = [{"kind": "capture", "race_id": "fixture"}]
    original.complete({"status": "PASS"}, elapsed=90, overrun=False)
    with pytest.raises(ValueError, match="identity_changed"):
        checkpoint(tmp_path, identity="other-source")
    resumed = checkpoint(tmp_path)
    assert resumed.value == original.value
    result_path = tmp_path / "evidence/phase-0-result.json"
    before = result_path.read_bytes()
    resumed.finish("DEFERRED")
    assert json.loads((tmp_path / "evidence/phase-checkpoint.json").read_text())["pending"]
    assert result_path.read_bytes() == before


def test_completed_evidence_cannot_be_overwritten(tmp_path):
    original = checkpoint(tmp_path)
    original.begin("capture", {}, "2026-07-19T12:00:00+10:00")
    original.complete({"status": "FAIL"}, elapsed=51, overrun=True)
    with pytest.raises(ValueError, match="not_started"):
        original.complete({"status": "PASS"}, elapsed=1, overrun=False)


def test_resume_rejects_changed_retained_result(tmp_path):
    original = checkpoint(tmp_path)
    original.begin("refresh", {}, "2026-07-19T12:00:00+10:00")
    original.complete({"status": "PASS"}, elapsed=1, overrun=False)
    (tmp_path / "evidence/phase-0-result.json").write_text("{}")
    with pytest.raises(ValueError, match="retained_result_changed"):
        checkpoint(tmp_path)


@pytest.mark.parametrize(
    "result,overrun", [({"status": "FAIL"}, False), ({"status": "PASS"}, True)]
)
def test_crash_after_failed_completion_cannot_turn_failure_into_success(tmp_path, result, overrun):
    original = checkpoint(tmp_path)
    original.begin("capture", {}, "2026-07-19T12:00:00+10:00")
    original.complete(result, elapsed=51, overrun=overrun)
    with pytest.raises(ValueError, match="failed_boundary_requires_reconciliation"):
        checkpoint(tmp_path)


def test_terminal_checkpoint_cannot_start_more_work(tmp_path):
    original = checkpoint(tmp_path)
    original.finish("LIVE_COLLECTION_COMPLETE")
    with pytest.raises(ValueError, match="terminal_checkpoint"):
        original.begin("refresh", {}, "2026-07-19T12:00:00+10:00")
