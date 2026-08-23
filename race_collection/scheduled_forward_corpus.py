"""Admit exact scheduled collector captures into the Phase 7 forward corpus."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from race_collection.domain import ArtifactChecksum, EvidenceField
from race_collection.forward_sealed_corpus import (
    PREJUMP_SOURCE,
    ForwardCorpusRejected,
    ForwardSealedCorpus,
    canonical_json,
)
from race_collection.manual_prediction_collector_request import (
    PROTOCOL_DIRECTORY,
    ManualPredictionCollectorProtocol,
    ProtocolRejected,
)
from race_collection.manual_prediction_collector_request import (
    canonical_bytes as protocol_canonical_bytes,
)
from race_collection.synchronous_manual_capture import (
    VerifiedCurrentRaceIndex,
    acquire_collector_lock_no_steal,
    release_owned_collector_lock,
)
from src.predictor.on_demand import sha256_bytes
from utils.runner_completeness import normalise_runner_name, parse_runner_rows_from_csv

ADMISSION_SCHEMA = "scheduled-forward-corpus-admission-v1"
FEATURE_BUNDLE_ID = "official-result-first-observation-v1-natural-canary"
MAX_EXACT_RECEIPT_AGE_SECONDS = 7 * 24 * 60 * 60
READY_CONTEXT_FEATURES = (
    ("canonical_race_identity", "race_identity", EvidenceField.RACE_IDENTITY.value),
    ("canonical_runner_identity", "runner_identity", EvidenceField.RUNNER_IDENTITY.value),
    ("venue", "race_card", EvidenceField.VENUE.value),
    ("distance", "race_card", EvidenceField.DISTANCE.value),
)
EXCLUDED_BASELINE_FEATURES = (
    ("race_card_context", "race_card"),
    ("form_context", "form"),
    ("speed_context", "speed"),
)
FORWARD_CAPTURE_FEATURES = (
    ("recent_workload", "runner_history"),
    ("prior_official_weight", "runner_history"),
    ("pir_running_position", "runner_history"),
    ("typed_trial_state", "trials"),
    ("steward_veterinary_state", "steward_veterinary"),
    ("lifecycle_age", "lifecycle"),
    ("sportsbet_win_market", "market"),
)
CANDIDATE_FEATURES = (
    tuple((name, family) for name, family, _field in READY_CONTEXT_FEATURES)
    + EXCLUDED_BASELINE_FEATURES
    + FORWARD_CAPTURE_FEATURES
)

FEATURE_SCHEMA_BYTES = canonical_json(
    {
        "bundle_id": FEATURE_BUNDLE_ID,
        "contract_version": "sealed-race-features-v1",
        "evidence_schema_version": "race-evidence-v1",
        "availability_manifest_version": "feature-availability-manifest-v1",
        "source_contracts": {
            PREJUMP_SOURCE: {
                "schema_versions": ["collector-canonical-runner-alignment-v1"],
                "provider_publication_time_exposed": False,
            }
        },
        "candidate_features": [
            {"name": name, "family": family, "semantics": "optional"}
            for name, family in CANDIDATE_FEATURES
        ],
        "fields": [
            {
                "name": "box_number",
                "family": "race_card",
                "semantics": "forecast-required",
                "source_field": "runner_features",
            }
        ],
        "normalization_version": "collector-canonical-runner-alignment-v1",
    }
)
MISSINGNESS_POLICY_BYTES = canonical_json(
    {
        "bundle_id": FEATURE_BUNDLE_ID,
        "feature_contract_version": "sealed-race-features-v1",
        "imputation": {},
    }
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ForwardCorpusRejected(f"{name} is missing or malformed")
    return value


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ForwardCorpusRejected(f"{name} is missing or ambiguous")
    return value


def _aware(value: Any, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(_text(value, name))
    except ValueError as error:
        raise ForwardCorpusRejected(f"{name} is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ForwardCorpusRejected(f"{name} must be timezone-aware")
    return parsed


def _distance_metres(value: Any) -> int:
    text = _text(value, "race distance").casefold()
    if text.endswith("m"):
        text = text[:-1]
    if not text.isascii() or not text.isdecimal() or int(text) <= 0:
        raise ForwardCorpusRejected("race distance is invalid")
    return int(text)


def _sha256(value: Any, name: str) -> str:
    text = _text(value, name)
    if text.startswith("sha256:"):
        text = text.removeprefix("sha256:")
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ForwardCorpusRejected(f"{name} is not a SHA-256 digest")
    return text


def _exact_json(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ForwardCorpusRejected(f"{name} is not valid JSON") from error
    if type(value) is not dict:
        raise ForwardCorpusRejected(f"{name} must be one JSON object")
    return value


def _validated_root(corpus_root: Path, evidence_root: Path) -> Path:
    if not corpus_root.is_absolute() or corpus_root.is_symlink():
        raise ForwardCorpusRejected("forward corpus root is ambiguous")
    root = corpus_root.resolve()
    evidence = evidence_root.resolve()
    if root == evidence or not root.is_relative_to(evidence) or not root.is_dir():
        raise ForwardCorpusRejected("forward corpus root is outside the evidence root")
    for name in ("artifacts", "races"):
        child = root / name
        if child.is_symlink() or not child.is_dir():
            raise ForwardCorpusRejected("forward corpus root is not initialized")
    return root


def _safe_raw_export(
    sidecar: Mapping[str, Any],
    *,
    evidence_root: Path,
    sidecar_path: Path,
) -> tuple[Path, bytes]:
    raw_path = Path(_text(sidecar.get("raw_export_path"), "raw_export_path"))
    if not raw_path.is_absolute() or raw_path.is_symlink() or not raw_path.is_file():
        raise ForwardCorpusRejected("raw source path is unsafe or missing")
    resolved = raw_path.resolve()
    evidence = evidence_root.resolve()
    if not resolved.is_relative_to(evidence):
        raise ForwardCorpusRejected("raw source escapes the evidence root")
    if resolved.parent != sidecar_path.parent / "raw_exports":
        raise ForwardCorpusRejected(
            "raw source is outside the canonical raw_exports lane"
        )
    raw = resolved.read_bytes()
    if sha256_bytes(raw) != _sha256(
        sidecar.get("raw_content_sha256"), "raw source hash"
    ) or len(raw) != sidecar.get("raw_content_length"):
        raise ForwardCorpusRejected("raw source bytes disagree with the sealed sidecar")
    return resolved, raw


def _canonical_source_url(value: Any) -> str:
    source = _text(value, "TheDogs source URL")
    parsed = urlsplit(source)
    if parsed.query not in {"", "trial=false"} or parsed.fragment:
        raise ForwardCorpusRejected("TheDogs source URL is not canonical")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _runner_rows(
    plan_item: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    raw_path: Path,
    verified_race: Mapping[str, Any],
) -> list[dict[str, Any]]:
    validation = _mapping(plan_item.get("runner_set_validation"), "runner validation")
    if validation.get("status") != "PASS":
        raise ForwardCorpusRejected("scheduled runner validation did not pass")
    expected = plan_item.get("expected_runners")
    if type(expected) is not list or len(expected) < 2:
        raise ForwardCorpusRejected("scheduled runners are missing")
    normalized: list[dict[str, Any]] = []
    seen_boxes: set[int] = set()
    seen_identities: set[str] = set()
    for value in expected:
        row = _mapping(value, "scheduled runner")
        if set(row) != {"box_number", "dog_name", "identity"}:
            raise ForwardCorpusRejected("scheduled runner identity envelope is invalid")
        box = row.get("box_number")
        name = _text(row.get("dog_name"), "scheduled runner name")
        identity = _text(row.get("identity"), "scheduled runner identity")
        if (
            type(box) is not int
            or not 1 <= box <= 20
            or identity != normalise_runner_name(name)
            or box in seen_boxes
            or identity in seen_identities
        ):
            raise ForwardCorpusRejected("scheduled runner identity is ambiguous")
        seen_boxes.add(box)
        seen_identities.add(identity)
        normalized.append({"box_number": box, "dog_name": name, "identity": identity})
    normalized.sort(key=lambda row: (row["box_number"], row["identity"]))
    if validation.get("expected_runners") not in (None, normalized):
        raise ForwardCorpusRejected("scheduled runner validation disagrees")

    shadow = _mapping(sidecar.get("prejump_shadow_metadata"), "pre-jump metadata")
    shadow_rows = shadow.get("runner_box_name_list")
    sidecar_rows = [
        {"box_number": row["box_number"], "dog_name": row["dog_name"]}
        for row in normalized
    ]
    if shadow_rows != sidecar_rows:
        raise ForwardCorpusRejected("pre-jump sidecar runner set disagrees")
    completeness = _mapping(
        sidecar.get("runner_completeness_after_canonical_alignment"),
        "canonical runner completeness",
    )
    if (
        completeness.get("status") != "COMPLETE"
        or completeness.get("participants") != sidecar_rows
    ):
        raise ForwardCorpusRejected(
            "canonical raw-source runner completeness disagrees"
        )
    raw_rows = parse_runner_rows_from_csv(raw_path)
    raw_identity = sorted(
        (row.box_number, normalise_runner_name(row.dog_name)) for row in raw_rows
    )
    expected_identity = sorted(
        (row["box_number"], row["identity"]) for row in normalized
    )
    if raw_identity != expected_identity:
        raise ForwardCorpusRejected(
            "raw source and scheduled runner identities disagree"
        )
    verified = verified_race.get("runners")
    if type(verified) is not list or len(verified) != len(normalized):
        raise ForwardCorpusRejected("verified current-index runner set disagrees")
    verified_rows: list[dict[str, Any]] = []
    verified_native_ids: set[str] = set()
    verified_boxes: set[int] = set()
    for value in verified:
        row = _mapping(value, "verified current-index runner")
        box = row.get("box")
        name = row.get("display_name")
        native_id = row.get("source_native_runner_id")
        if (
            type(box) is not int
            or type(name) is not str
            or not name.strip()
            or row.get("scratch_state") != "ACTIVE"
            or type(native_id) is not str
            or not native_id.isascii()
            or not native_id.isdecimal()
        ):
            raise ForwardCorpusRejected(
                "verified current-index requires numeric native runner IDs"
            )
        if native_id in verified_native_ids or box in verified_boxes:
            raise ForwardCorpusRejected("verified current-index runner set is ambiguous")
        verified_native_ids.add(native_id)
        verified_boxes.add(box)
        verified_rows.append(
            {
                "source_native_runner_id": native_id,
                "name": name.strip(),
                "box_number": box,
            }
        )
    return verified_rows


def _verified_race(
    verified_index: VerifiedCurrentRaceIndex | None,
    plan_item: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        not isinstance(verified_index, VerifiedCurrentRaceIndex)
        or verified_index.schema_version != "collector_current_race_index_v2"
    ):
        raise ForwardCorpusRejected("verified collector current-race index is required")
    matches = [
        race
        for race in verified_index.races
        if isinstance(race, Mapping) and race.get("race_id") == plan_item.get("race_id")
    ]
    if len(matches) != 1:
        raise ForwardCorpusRejected("scheduled race is absent from verified current index")
    race = matches[0]
    native_race_id = race.get("source_native_race_id")
    if (
        race.get("date") != plan_item.get("race_date")
        or race.get("venue") != plan_item.get("venue")
        or race.get("race_number") != plan_item.get("race_number")
        or race.get("jump_datetime") != plan_item.get("jump_datetime")
        or race.get("race_url") != plan_item.get("thedogs_source_url")
        or type(native_race_id) is not str
        or not native_race_id.isascii()
        or not native_race_id.isdecimal()
        or type(race.get("distance_metres")) is not int
        or race["distance_metres"] <= 0
    ):
        raise ForwardCorpusRejected("verified current-index race identity disagrees")
    return race


def _scheduled_handoff(
    *,
    protocol: ManualPredictionCollectorProtocol,
    plan_item: Mapping[str, Any],
    attempt: Mapping[str, Any] | None,
    receipt_publish: Mapping[str, Any] | None,
    collector_run_id: str,
    emitted_at: datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if plan_item.get("schema_version") != "autonomous_live_odds_capture_plan_item_v1":
        raise ForwardCorpusRejected("manual or unsupported capture plan rejected")
    if plan_item.get("status") != "READY_TO_CAPTURE" or not plan_item.get(
        "sidecar_path"
    ):
        raise ForwardCorpusRejected("scheduled capture is not corpus-eligible")
    race_id = _text(plan_item.get("race_id"), "scheduled race identity")
    published_hash: str | None = None
    if receipt_publish is not None:
        if (
            receipt_publish.get("schema_version")
            != "collector_exact_capture_receipt_publish_v1"
            or receipt_publish.get("status") != "PUBLISHED"
            or receipt_publish.get("source_race_id") != race_id
        ):
            raise ForwardCorpusRejected("exact scheduled receipt was not published")
        matches = [
            row
            for row in receipt_publish.get("receipts") or []
            if isinstance(row, Mapping) and row.get("race_id") == race_id
        ]
        if len(matches) != 1:
            raise ForwardCorpusRejected("exact source-race receipt is ambiguous")
        published_hash = _sha256(
            matches[0].get("capture_attempt_sha256"),
            "published capture attempt hash",
        )
    try:
        handoff = protocol.discover_collector_exact_handoff(
            race_id=race_id,
            current_time=emitted_at,
            max_age_seconds=MAX_EXACT_RECEIPT_AGE_SECONDS,
        )
    except ProtocolRejected as error:
        raise ForwardCorpusRejected(f"scheduled receipt rejected: {error}") from error
    if handoff is None:
        raise ForwardCorpusRejected("exact scheduled receipt is unavailable")
    if (
        published_hash is not None
        and handoff.get("capture_attempt_sha256") != published_hash
    ):
        raise ForwardCorpusRejected(
            "published and discovered capture receipts disagree"
        )
    report = _exact_json(handoff["_report_bytes"], "scheduled source report")
    sealed_collector_run_id = _text(
        report.get("collector_run_id"), "sealed scheduled collector authority"
    )
    if (
        attempt is not None or receipt_publish is not None
    ) and sealed_collector_run_id != collector_run_id:
        raise ForwardCorpusRejected("scheduled collector authority disagrees")
    sealed_plan = _mapping(report.get("source_plan_item"), "sealed scheduled plan")
    sealed_attempt = _mapping(report.get("source_attempt"), "sealed scheduled attempt")
    if (
        sealed_plan.get("schema_version") != "autonomous_live_odds_capture_plan_item_v1"
        or sealed_plan.get("status") != "READY_TO_CAPTURE"
    ):
        raise ForwardCorpusRejected("sealed scheduled plan is not corpus-eligible")
    if attempt is not None:
        if canonical_json(dict(sealed_plan)) != canonical_json(dict(plan_item)):
            raise ForwardCorpusRejected("scheduled plan identity or source drift")
        if canonical_json(dict(sealed_attempt)) != canonical_json(dict(attempt)):
            raise ForwardCorpusRejected("scheduled capture attempt drift")
    else:
        immutable_fields = (
            "race_id",
            "venue",
            "race_number",
            "race_date",
            "race_time",
            "jump_datetime",
            "thedogs_source_url",
            "expected_runners",
        )
        if any(
            plan_item.get(field) != sealed_plan.get(field) for field in immutable_fields
        ):
            raise ForwardCorpusRejected("scheduled replay identity or source drift")
        current_validation = _mapping(
            plan_item.get("runner_set_validation"), "current runner validation"
        )
        sealed_validation = _mapping(
            sealed_plan.get("runner_set_validation"), "sealed runner validation"
        )
        if (
            current_validation.get("status") != "PASS"
            or sealed_validation.get("status") != "PASS"
            or current_validation.get("expected_runners")
            != sealed_validation.get("expected_runners")
            or plan_item.get("blockers") not in (None, [])
            or sealed_plan.get("blockers") not in (None, [])
        ):
            raise ForwardCorpusRejected("scheduled replay runner or blocker drift")
    if sha256_bytes(protocol_canonical_bytes(dict(sealed_attempt))) != handoff.get(
        "capture_attempt_sha256"
    ):
        raise ForwardCorpusRejected("scheduled capture attempt hash drift")
    return handoff, dict(sealed_plan)


def admit_scheduled_capture(
    *,
    protocol: ManualPredictionCollectorProtocol,
    evidence_root: Path,
    corpus_root: Path,
    collector_run_id: str,
    plan_item: Mapping[str, Any],
    verified_index: VerifiedCurrentRaceIndex | None = None,
    emitted_at: datetime,
    cohort_id: str | None = None,
    cohort_checksum: ArtifactChecksum | None = None,
    attempt: Mapping[str, Any] | None = None,
    receipt_publish: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one exact scheduled receipt and append/replay its pre-jump stage."""

    if emitted_at.tzinfo is None or emitted_at.utcoffset() is None:
        raise ForwardCorpusRejected("admission timestamp must be timezone-aware")
    evidence = evidence_root.resolve()
    expected_protocol_root = evidence / PROTOCOL_DIRECTORY
    if (
        protocol.root.is_symlink()
        or protocol.root.resolve() != expected_protocol_root.resolve()
    ):
        raise ForwardCorpusRejected("scheduled receipt protocol root is ambiguous")
    root = _validated_root(corpus_root, evidence)
    collector_run_id = _text(collector_run_id, "scheduled collector run identity")
    verified_race = _verified_race(verified_index, plan_item)
    handoff, sealed_plan_item = _scheduled_handoff(
        protocol=protocol,
        plan_item=plan_item,
        attempt=attempt,
        receipt_publish=receipt_publish,
        collector_run_id=collector_run_id,
        emitted_at=emitted_at,
    )
    plan_item = sealed_plan_item
    sidecar_path = Path(handoff["_sidecar_path"])
    form_path = Path(handoff["_form_path"])
    if (
        sidecar_path
        != Path(_text(plan_item.get("sidecar_path"), "sidecar path")).resolve()
        or form_path != Path(_text(plan_item.get("csv_path"), "form path")).resolve()
    ):
        raise ForwardCorpusRejected("scheduled source paths disagree")
    sidecar = _exact_json(handoff["_sidecar_bytes"], "scheduled source sidecar")
    if (
        sidecar.get("normalization_status") != "verified"
        or sidecar.get("metadata_is_leakage_safe") is not True
        or Path(_text(sidecar.get("accepted_csv_path"), "accepted form path")).resolve()
        != form_path
        or sha256_bytes(handoff["_form_bytes"])
        != _sha256(sidecar.get("content_sha256"), "accepted form hash")
    ):
        raise ForwardCorpusRejected("scheduled normalized source binding is invalid")
    raw_path, _retained_scheduled_source = _safe_raw_export(
        sidecar,
        evidence_root=evidence,
        sidecar_path=sidecar_path,
    )
    shadow = _mapping(sidecar.get("prejump_shadow_metadata"), "pre-jump metadata")
    alignment = _mapping(
        shadow.get("canonical_final_runner_alignment"),
        "canonical final runner alignment",
    )
    canonical_alignment = _mapping(
        sidecar.get("canonical_runner_alignment"),
        "canonical runner alignment",
    )
    if (
        shadow.get("status") != "PASS"
        or shadow.get("metadata_is_leakage_safe") is not True
        or shadow.get("fail_reasons") not in (None, [])
        or alignment.get("status") != "aligned"
        or alignment.get("canonical_runner_set_status") != "available"
        or canonical_alignment.get("status") != "aligned"
        or canonical_alignment.get("missing_canonical_participants") not in (None, [])
        or canonical_alignment.get("dropped_participants") not in (None, [])
    ):
        raise ForwardCorpusRejected("scheduled source alignment or leakage gate failed")
    source_url = _text(plan_item.get("thedogs_source_url"), "scheduled source URL")
    if any(
        value not in (None, source_url)
        for value in (
            sidecar.get("race_url"),
            shadow.get("source_url"),
            canonical_alignment.get("canonical_source_url"),
        )
    ):
        raise ForwardCorpusRejected("scheduled source URL drift")
    canonical_url = _canonical_source_url(source_url)
    race_id = _text(plan_item.get("race_id"), "scheduled race identity")
    corpus = ForwardSealedCorpus(root, clock=lambda: emitted_at)
    frozen_race_id = corpus.frozen_race_id_for_source_native_race(
        verified_race["source_native_race_id"],
        cohort_id=cohort_id,
        expected_cohort_checksum=cohort_checksum,
        expected_distance_metres=verified_race["distance_metres"],
    )
    if frozen_race_id is not None:
        race_id = frozen_race_id
    racing_date = _text(plan_item.get("race_date"), "racing date")
    jump_text = _text(plan_item.get("jump_datetime"), "scheduled jump")
    observed_text = _text(
        shadow.get("metadata_captured_at") or sidecar.get("metadata_captured_at"),
        "source observed timestamp",
    )
    frozen_text = _text(handoff.get("append_timestamp"), "feature freeze timestamp")
    observed = _aware(observed_text, "source observed timestamp")
    frozen = _aware(frozen_text, "feature freeze timestamp")
    jump = _aware(jump_text, "scheduled jump")
    if not observed <= frozen < jump:
        raise ForwardCorpusRejected(
            "scheduled source/feature timing is not prospective"
        )
    if (
        shadow.get("race_date") != racing_date
        or str(shadow.get("race_number")) != str(plan_item.get("race_number"))
        or shadow.get("venue") != plan_item.get("venue")
    ):
        raise ForwardCorpusRejected("scheduled race identity disagrees with sidecar")
    runners = _runner_rows(plan_item, sidecar, raw_path, verified_race)
    raw_source = verified_index.packet_bytes
    if (
        type(raw_source) is not bytes
        or not raw_source
        or sha256_bytes(raw_source) != verified_index.packet_sha256
    ):
        raise ForwardCorpusRejected("verified current-index packet bytes have hash drift")
    runner_ids = sorted(
        (row["source_native_runner_id"] for row in runners),
        key=lambda value: unicodedata.normalize(
            "NFKC", " ".join(value.split())
        ).casefold(),
    )
    runner_features = {
        row["source_native_runner_id"]: {"box_number": row["box_number"]}
        for row in runners
    }
    race_info = _mapping(sidecar.get("race_info"), "race metadata")
    distance_metres = _distance_metres(race_info.get("distance") or shadow.get("distance"))
    if distance_metres != verified_race["distance_metres"]:
        raise ForwardCorpusRejected("verified current-index race distance disagrees")
    raw_checksum = "sha256:" + sha256_bytes(raw_source)
    fields = {
        "runner_set": runner_ids,
        "runner_identity": {runner_id: "authoritative" for runner_id in runner_ids},
        "runner_features": runner_features,
        "race_identity": verified_race["source_native_race_id"],
        "venue": plan_item["venue"],
        "distance": distance_metres,
        "feature_availability": {
            "box_number": {
                "status": "READY_NOW",
                "source_name": PREJUMP_SOURCE,
                "source_schema_version": "collector-canonical-runner-alignment-v1",
                "source_native_race_id": verified_race["source_native_race_id"],
                "source_native_runner_ids": runner_ids,
                "provider_published_at": None,
                "collector_received_at": observed_text,
                "completeness": "COMPLETE",
                "whole_race_coverage": True,
                "derivation_version": "scheduled-forward-box-number-v1",
                "blocking_reasons": [],
            },
            **{
                name: {
                    "status": "READY_NOW",
                    "source_name": PREJUMP_SOURCE,
                    "source_schema_version": "collector-canonical-runner-alignment-v1",
                    "source_native_race_id": verified_race["source_native_race_id"],
                    "source_native_runner_ids": runner_ids,
                    "provider_published_at": None,
                    "collector_received_at": observed_text,
                    "completeness": "COMPLETE",
                    "whole_race_coverage": True,
                    "derivation_version": f"scheduled-forward-{name}-v1",
                    "blocking_reasons": [],
                }
                for name, _family, _field in READY_CONTEXT_FEATURES
            },
            **{
                name: {
                    "status": (
                        "EXCLUDED"
                        if (name, _family) in EXCLUDED_BASELINE_FEATURES
                        else "FORWARD_CAPTURE"
                    ),
                    "source_name": None,
                    "source_schema_version": None,
                    "source_native_race_id": None,
                    "source_native_runner_ids": [],
                    "provider_published_at": None,
                    "collector_received_at": None,
                    "completeness": "UNKNOWN",
                    "whole_race_coverage": False,
                    "derivation_version": f"{name}-unavailable-v1",
                    "blocking_reasons": [
                        "SOURCE_UNAVAILABLE",
                        "INCOMPLETE_COVERAGE",
                        "NORMALIZED_EVIDENCE_MISSING",
                        *(
                            ["NOT_REQUESTED_BY_BASELINE"]
                            if (name, _family) in EXCLUDED_BASELINE_FEATURES
                            else (
                            ["SOURCE_AUTHORIZATION_REQUIRED"]
                            if name != "sportsbet_win_market"
                            else []
                            )
                        ),
                    ],
                }
                for name, _family in (EXCLUDED_BASELINE_FEATURES + FORWARD_CAPTURE_FEATURES)
            },
        },
    }
    sealed_evidence = canonical_json(
        {
            "schema_version": "race-evidence-v1",
            "normalization_version": "collector-canonical-runner-alignment-v1",
            "race_id": race_id,
            "fields": fields,
            "field_provenance": [
                {
                    "field": field,
                    "authority": "canonical-thedogs-final-runner-set",
                    "critical": EvidenceField(field).critical,
                    "value": fields[field],
                    "source": PREJUMP_SOURCE,
                    "artifact_checksum": raw_checksum,
                }
                for field in (
                    "runner_set",
                    "runner_identity",
                    "runner_features",
                    "race_identity",
                    "venue",
                    "distance",
                )
            ],
            "freeze": {
                "at": frozen_text,
                "authority": "scheduled-canonical-live-odds-capture-v1",
                "odds_checksum": "sha256:" + handoff["capture_attempt_sha256"],
            },
        }
    )
    lock = acquire_collector_lock_no_steal(
        root / "forward-sealed-corpus.lock",
        run_id=f"forward_corpus_{collector_run_id}",
        output_dir=root,
        phase="scheduled_forward_corpus_admission",
        acquisition_policy="scheduled_forward_corpus_no_steal_v1",
    )
    try:
        existing = corpus._load_receipt(race_id, "prejump")
        receipt = corpus.capture_prejump(
            race_id=race_id,
            racing_date=racing_date,
            raw_source_bytes=raw_source,
            sealed_evidence_bytes=sealed_evidence,
            feature_schema_bytes=FEATURE_SCHEMA_BYTES,
            missingness_policy_bytes=MISSINGNESS_POLICY_BYTES,
            source_name=PREJUMP_SOURCE,
            canonical_source_url=canonical_url,
            source_native_race_id=verified_race["source_native_race_id"],
            runners=runners,
            meeting_metadata={
                "meeting_code": plan_item.get("venue"),
                "venue": plan_item.get("venue"),
            },
            race_metadata={
                "distance_metres": distance_metres,
                "race_number": plan_item.get("race_number"),
                "race_time": plan_item.get("race_time"),
            },
            source_observed_at=observed_text,
            feature_frozen_at=frozen_text,
            scheduled_jump_at=jump_text,
        )
    finally:
        release_owned_collector_lock(lock)
    return {
        "schema_version": ADMISSION_SCHEMA,
        "status": "EXACT_REPLAY" if existing is not None else "PREJUMP_CAPTURED",
        "race_id": race_id,
        "capture_attempt_sha256": handoff["capture_attempt_sha256"],
        "raw_source_sha256": sha256_bytes(raw_source),
        "prejump_receipt_sha256": sha256_bytes(canonical_json(receipt)),
    }
