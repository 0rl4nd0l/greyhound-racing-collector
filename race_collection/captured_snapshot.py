"""Prove a consumed capture was superseded; never rebind it for prediction."""
from datetime import datetime
import json
from pathlib import Path

from race_collection.manual_prediction_collector_request import (
    canonical_bytes, runner_set_sha256, sha256_bytes,
)
from race_collection.synchronous_manual_capture import (
    CURRENT_RACE_INDEX_SCHEMA, MAX_CURRENT_INDEX_RACES, VerifiedCurrentRaceIndex,
    _RetainedSafeFiles, _normalize_current_index_rows, _v2_runner_rows,
)


def classify_superseded_capture(reserved, view, *, evidence_root, current_time):
    """Authenticate both snapshots before allowing a pre-job race exclusion.

    The claim has already been authenticated against the campaign ledger and
    exact capture receipt. Its packet hash anchors this historical replay. The
    ordinary reader must have verified the current view; it stays authoritative
    for freshness. No input, receipt, claim or publication is rewritten.
    """
    if not isinstance(view, VerifiedCurrentRaceIndex):
        raise ValueError("capture_supersession_current_view_unverified")
    generated = datetime.fromisoformat(view.source_generated_at)
    if not 0 <= (current_time - generated).total_seconds() <= 300:
        raise ValueError("CURRENT_INDEX_STALE")
    item = reserved["item"]
    identity = item["race_identity"]
    root = Path(evidence_root).absolute()
    paths = [Path(path) for path in item["input_files"]]
    # Native full/odds publications keep their inputs below one phase directory.
    phases = {root / path.absolute().relative_to(root).parts[0] for path in paths}
    if len(paths) != 2 or len(phases) != 1:
        raise ValueError("capture_supersession_source_ambiguous")
    with _RetainedSafeFiles(root) as retained:
        publication_raw = retained.read(next(iter(phases)) / "current_race_index_publish.json",
            missing_code="CAPTURE_PUBLICATION_MISSING")
        publication = json.loads(publication_raw)
        if (publication.get("schema_version") != "collector_current_race_index_publish_v2"
                or publication.get("status") != "PUBLISHED"
                or publication.get("packet_sha256") != item["packet_sha256"]):
            raise ValueError("capture_supersession_publication_changed")
        source_path = Path(publication["source_refresh_report_path"])
        source_path = source_path if source_path.is_absolute() else root / source_path
        raw = retained.read(source_path, missing_code="CAPTURE_REFRESH_SOURCE_MISSING")
        source = json.loads(raw)
        if (sha256_bytes(raw) != publication["source_refresh_report_sha256"]
                or source.get("status") != "SUCCESS" or source.get("dry_run") is True):
            raise ValueError("capture_supersession_refresh_changed")
        races = []
        for race in _normalize_current_index_rows(source, max_races=MAX_CURRENT_INDEX_RACES):
            runners, provenance, digest = _v2_runner_rows(
                race, source, evidence_root=root, snapshot=retained)
            races.append({**race, "runners": runners, "runner_source": provenance,
                          "runner_set_sha256": digest})
        old_generated = datetime.fromisoformat(source["generated_at"])
        packet = dict(schema_version=CURRENT_RACE_INDEX_SCHEMA, run_id=publication["run_id"],
            source_generated_at=old_generated.isoformat(),
            source_refresh_report_path=source_path.absolute().relative_to(root).as_posix(),
            source_refresh_report_sha256=sha256_bytes(raw), race_count=len(races),
            max_races=MAX_CURRENT_INDEX_RACES, races=races)
        if sha256_bytes(canonical_bytes(packet)) != item["packet_sha256"]:
            raise ValueError("capture_supersession_packet_changed")
        old = [race for race in races if race["race_id"] == item["race_id"]]
        new = [race for race in view.races if race["race_id"] == item["race_id"]]
        if len(old) != 1 or len(new) != 1:
            raise ValueError("capture_supersession_race_ambiguous")
        old, new = old[0], new[0]
        fields = ("race_id", "race_url", "jump_datetime", "source_native_race_id")
        if (any(old[key] != identity[key] or new[key] != identity[key] for key in fields)
                or any(old[key] != new[key] for key in ("date", "venue", "race_number"))
                or old["runner_set_sha256"] != identity["runner_set_sha256"]
                or new["runner_set_sha256"] == identity["runner_set_sha256"]
                or not old_generated < generated <= current_time
                or current_time >= datetime.fromisoformat(identity["jump_datetime"])):
            raise ValueError("capture_supersession_identity_or_time_changed")
        provenance = old["runner_source"]
        expected_files = {
            str(root / provenance["csv_path"]): provenance["csv_sha256"],
            str(root / provenance["sidecar_path"]): provenance["sidecar_sha256"],
        }
        expected_runners = [{"box_number": row["box"], "dog_name": row["display_name"],
                             "identity": row["identity"]} for row in old["runners"]]
        if (expected_files != item["input_files"]
                or runner_set_sha256(expected_runners) != item["capture_runner_set_sha256"]):
            raise ValueError("capture_supersession_reserved_inputs_changed")
        retained.validate()
    roster_fields = ("box", "identity", "source_native_runner_id", "scratch_state")
    roster = lambda race: [tuple(row[key] for key in roster_fields) for row in race["runners"]]
    same = roster(old) == roster(new)
    return {"schema_version": "verified_capture_supersession_v1",
        "code": "CAPTURE_SNAPSHOT_SUPERSEDED" if same else "CAPTURE_RUNNERS_CHANGED",
        "race_id": item["race_id"], "same_roster": same,
        "captured_packet_sha256": item["packet_sha256"], "current_packet_sha256": view.packet_sha256,
        "captured_runner_set_sha256": old["runner_set_sha256"],
        "current_runner_set_sha256": new["runner_set_sha256"],
        "captured_runner_count": len(old["runners"]), "current_runner_count": len(new["runners"]),
        "captured_publication_sha256": sha256_bytes(publication_raw),
        "captured_source_generated_at": old_generated.isoformat(),
        "current_source_generated_at": generated.isoformat(),
        "checked_at": current_time.isoformat(), "capture_consumed": True,
        "job_created": False, "comparison_admitted": False}
