#!/usr/bin/env python3
"""Build the four hash-bound Greyhound Semantic Anti-Loop V2 seed decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SEED_VALIDATED_AT = "2026-07-13T18:00:00+10:00"
PINNED_HASHES = {
    "floor": "f878b67628e8f462f3aa7578b9a65e011a11a0b738479f6e18c2e22dffb786dd",
    "evaluation": "e935c9c65dafd1355cedf24fd0aaf646b800bc92d41a682d6d4ea2c35c1d5da6",
    "strict_overlap": "870067e6f4024647162265ebcf850850a855c3314e6c5c1e008627d0624f3b85",
    "bridge": "bd5557dc66a2c29e50e674551e84ec0bdac5baaf91e412e61a05f9b2a6a67efd",
}
SCOPE_FIELDS = (
    "project_id",
    "claim_id",
    "hypothesis_id",
    "source_class",
    "dataset_version",
    "evidence_hash",
    "target_transition",
)
BRIDGE_WRITE_KEYS = {
    "canonical_database",
    "database_copy_apply",
    "reader_deployment",
    "runtime",
}


class SeedEvidenceError(ValueError):
    """Raised when a recorded seed artifact no longer proves the seeded claim."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_bytes(path: Path, expected: str, label: str) -> tuple[bytes, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SeedEvidenceError(f"{label} is not readable: {exc}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise SeedEvidenceError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    return raw, actual


def _load_json(path: Path, expected_hash: str, label: str) -> tuple[dict[str, Any], str]:
    raw, digest = _verified_bytes(path, expected_hash, label)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SeedEvidenceError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SeedEvidenceError(f"{label} must contain one JSON object")
    return payload, digest


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SeedEvidenceError(message)


def compute_scope_fingerprint(entry: Mapping[str, Any]) -> str:
    values: list[str] = []
    for field in SCOPE_FIELDS:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SeedEvidenceError(f"{field} must be a non-empty string")
        normalized = value.strip()
        if field == "evidence_hash":
            normalized = normalized.lower()
            if normalized.startswith("sha256:"):
                normalized = normalized.removeprefix("sha256:")
            _require(
                len(normalized) == 64
                and all(character in "0123456789abcdef" for character in normalized),
                "evidence_hash must be a SHA-256 digest",
            )
            normalized = f"sha256:{normalized}"
        values.append(normalized)
    canonical = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def composite_evidence_hash(evaluation_hash: str, overlap_hash: str) -> str:
    """Bind strict Sportsbet readiness to its evaluation and overlap artifacts."""

    ordered: list[str] = []
    for label, digest in (
        ("evaluation", evaluation_hash),
        ("strict_overlap", overlap_hash),
    ):
        normalized = digest.strip().lower().removeprefix("sha256:")
        _require(
            len(normalized) == 64
            and all(character in "0123456789abcdef" for character in normalized),
            f"{label} hash must be a SHA-256 digest",
        )
        ordered.append(f"sha256:{normalized}")
    canonical = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _entry(**values: Any) -> dict[str, Any]:
    entry = {
        "project_id": "greyhound_racing_collector",
        "validated_at": SEED_VALIDATED_AT,
        **values,
    }
    entry["scope_fingerprint"] = compute_scope_fingerprint(entry)
    return entry


def _strict_overlap_counts(path: Path, expected_hash: str) -> tuple[int, int, str]:
    raw, digest = _verified_bytes(path, expected_hash, "strict Sportsbet overlap")
    try:
        text = raw.decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise SeedEvidenceError(f"strict Sportsbet overlap is not readable CSV: {exc}") from exc
    _require(
        "complete_strict_sportsbet_win_odds" in (rows[0] if rows else {}),
        "strict Sportsbet overlap is missing the completeness column",
    )
    complete = sum(
        str(row.get("complete_strict_sportsbet_win_odds", "")).strip().lower()
        == "true"
        for row in rows
    )
    return len(rows), complete, digest


def build_seed_entries(
    *,
    floor_summary: Path,
    evaluation_results: Path,
    strict_overlap: Path,
    bridge_proof: Path,
    expected_hashes: Mapping[str, str] = PINNED_HASHES,
) -> list[dict[str, Any]]:
    floor, floor_hash = _load_json(
        floor_summary, expected_hashes["floor"], "TheDogs floor summary"
    )
    _require(
        floor.get("schema_version") == "thedogs_published_market_history_audit_v1",
        "TheDogs floor summary has an unexpected schema",
    )
    _require(
        floor.get("source_class") == "thedogs_published_market_history",
        "TheDogs floor summary has an unexpected source class",
    )
    _require(
        floor.get("complete_thedogs_published_market_history_races") == 663
        and floor.get("required_floor") == 300
        and floor.get("meets_300_floor") is True,
        "TheDogs recorded snapshot does not prove the 663-race floor decision",
    )
    floor_writes = floor.get("no_write_guarantees")
    _require(
        isinstance(floor_writes, dict)
        and floor_writes.get("report_only") is True
        and floor_writes.get("db_writes") is False
        and floor_writes.get("registry_writes") is False,
        "TheDogs floor artifact does not preserve its report-only boundary",
    )

    evaluation, evaluation_hash = _load_json(
        evaluation_results,
        expected_hashes["evaluation"],
        "aggregate challenger evaluation",
    )
    decision = evaluation.get("decision")
    strict = evaluation.get("strict_sportsbet_baseline")
    floors = evaluation.get("floors_and_split")
    training = evaluation.get("training")
    _require(
        evaluation.get("schema_version")
        == "thedogs_published_market_large_csv_history_challenger_report_v2",
        "aggregate challenger evaluation has an unexpected schema",
    )
    _require(
        evaluation.get("final_status") == "KEEP_BASELINE"
        and isinstance(decision, dict)
        and decision.get("recommendation") == "KEEP_BASELINE"
        and decision.get("passes_all_acceptance_gates") is False
        and decision.get("qualifying_candidate_keys") == [],
        "aggregate challenger evaluation does not prove KEEP_BASELINE",
    )
    _require(
        isinstance(floors, dict)
        and floors.get("complete_races") == 663
        and floors.get("eval_races") == 300
        and isinstance(training, dict)
        and training.get("model_count") == 9,
        "aggregate challenger evaluation has an unexpected 663-race split or model count",
    )
    _require(
        isinstance(strict, dict)
        and strict.get("status") == "DATA_MISSING_FLOOR"
        and strict.get("complete_eval_overlap_races") == 135
        and strict.get("same_floor_overlap_cleared") is False
        and strict.get("kept_separate_from_published_market_gate") is True,
        "evaluation does not prove the separate strict Sportsbet DATA_MISSING floor",
    )

    overlap_rows, overlap_complete, overlap_hash = _strict_overlap_counts(
        strict_overlap, expected_hashes["strict_overlap"]
    )
    _require(
        overlap_rows == 663 and overlap_complete == 135,
        "strict Sportsbet overlap must contain 663 races with exactly 135 complete rows",
    )

    bridge, bridge_hash = _load_json(
        bridge_proof, expected_hashes["bridge"], "historical identity bridge proof"
    )
    bridge_counts = bridge.get("bridge")
    bridge_writes = bridge.get("writes_performed")
    reader_manifest = bridge.get("reader_manifest")
    _require(
        bridge.get("schema_version") == "historical_race_identity_bridge_proof_v1"
        and bridge.get("bridge_result") == "REPORT_ONLY_BRIDGE_READY"
        and bridge.get("result") == "COPY_REPAIR_BLOCKED",
        "identity bridge does not prove report-only readiness with copy repair blocked",
    )
    _require(
        isinstance(bridge_writes, dict)
        and set(bridge_writes) == BRIDGE_WRITE_KEYS
        and all(value is False for value in bridge_writes.values()),
        "identity bridge writes_performed must contain exactly the four no-write keys",
    )
    _require(
        isinstance(bridge_counts, dict)
        and bridge_counts.get("race_count") == 662
        and bridge_counts.get("snapshot_count") == 1872
        and isinstance(reader_manifest, dict)
        and len(reader_manifest.get("blockers", [])) == 3,
        "identity bridge proof has unexpected counts or copy-repair blockers",
    )

    floor_ref = str(floor_summary.resolve())
    evaluation_ref = str(evaluation_results.resolve())
    overlap_ref = str(strict_overlap.resolve())
    bridge_ref = str(bridge_proof.resolve())
    return [
        _entry(
            decision_id="greyhound-thedogs-floor-663-20260709",
            task_id="thedogs_published_market_history_20260709",
            run_id="thedogs_published_market_history_20260709T_report_only",
            claim_id="thedogs_historical_source_floor",
            hypothesis_id="thedogs_published_market_history_clears_300_race_floor",
            program_track="offline_development",
            source_class="thedogs_published_market_history",
            dataset_version="thedogs_published_market_history_20260709_663",
            evidence_hash=f"sha256:{floor_hash}",
            target_transition="clear_thedogs_historical_source_floor",
            phase_before="historical_source_floor_unproven",
            phase_after="historical_source_floor_cleared_at_663",
            decision="PASS",
            outcome_status="ADVANCED",
            decision_delta="The recorded report-only snapshot proves 663 complete TheDogs published-market races against the 300-race floor.",
            evidence_refs=[floor_ref],
            blocks=[],
            does_not_block=[
                "strict_sportsbet_same_floor_comparison",
                "offline_feature_research",
            ],
            invalidation_conditions=[
                "The artifact hash or source-class semantics changes.",
                "The 663-race snapshot is shown to include ineligible races.",
            ],
            reopen_conditions=[
                "A new dataset version or evidence hash is supplied.",
                "A genuinely new hypothesis tests a different floor or eligibility rule.",
            ],
        ),
        _entry(
            decision_id="greyhound-aggregate-challenger-keep-baseline-20260709",
            task_id="thedogs_published_market_large_csv_history_challenger_20260709",
            run_id="thedogs_published_market_large_csv_history_challenger_663",
            claim_id="historical_aggregate_challenger",
            hypothesis_id="aggregate_history_challenger_beats_market_only_baseline",
            program_track="offline_development",
            source_class="thedogs_published_market_history",
            dataset_version="thedogs_history_challenger_20260709_663",
            evidence_hash=f"sha256:{evaluation_hash}",
            target_transition="promote_historical_aggregate_challenger",
            phase_before="aggregate_challenger_unresolved",
            phase_after="keep_market_only_baseline",
            decision="FAIL",
            outcome_status="ADVANCED",
            decision_delta="Nine evaluated models produced no qualifying challenger; the recorded decision remained KEEP_BASELINE.",
            evidence_refs=[evaluation_ref],
            blocks=["promote_historical_aggregate_challenger"],
            does_not_block=[
                "offline_feature_research",
                "new_hypothesis_research_fit",
            ],
            invalidation_conditions=[
                "The evaluation artifact hash, split, acceptance gates, or source semantics changes."
            ],
            reopen_conditions=[
                "A new dataset/evidence version is evaluated.",
                "A genuinely new challenger hypothesis is declared.",
            ],
        ),
        _entry(
            decision_id="greyhound-strict-sportsbet-same-floor-data-missing-20260709",
            task_id="thedogs_published_market_large_csv_history_challenger_20260709",
            run_id="strict_sportsbet_overlap_recorded_snapshot_20260709",
            claim_id="strict_sportsbet_same_floor_readiness",
            hypothesis_id="strict_sportsbet_supports_same_floor_comparison",
            program_track="prospective_readiness",
            source_class="strict_prejump_sportsbet_win_odds",
            dataset_version="strict_sportsbet_overlap_recorded_snapshot_20260709",
            evidence_hash=f"sha256:{composite_evidence_hash(evaluation_hash, overlap_hash)}",
            target_transition="strict_sportsbet_same_floor_comparison",
            phase_before="strict_same_floor_evidence_unproven",
            phase_after="strict_same_floor_data_missing",
            decision="DATA_MISSING",
            outcome_status="DATA_MISSING",
            decision_delta="NO_DELTA",
            evidence_refs=[evaluation_ref, overlap_ref],
            blocks=[
                "strict_sportsbet_same_floor_comparison",
                "prospective_model_promotion",
            ],
            does_not_block=[
                "offline_research_fit",
                "offline_feature_research",
                "thedogs_historical_source_floor",
            ],
            invalidation_conditions=[
                "The evaluation or strict-overlap artifact hash, or eligibility semantics, changes."
            ],
            reopen_conditions=[
                "Fresh strict Sportsbet evidence changes the dataset/evidence hash and clears the declared floor.",
                "A genuinely new hypothesis targets a different prospective comparison.",
            ],
        ),
        _entry(
            decision_id="greyhound-historical-identity-bridge-report-only-ready-20260712",
            task_id="historical_race_repair_v1_20260711",
            run_id="race_identity_bridge_v1_20260712",
            claim_id="historical_identity_bridge",
            hypothesis_id="reviewed_bridge_supports_report_only_historical_reads",
            program_track="offline_development",
            source_class="historical_race_identity_bridge",
            dataset_version="race_identity_bridge_v1_20260712_662",
            evidence_hash=f"sha256:{bridge_hash}",
            target_transition="use_report_only_historical_identity_bridge",
            phase_before="historical_identity_bridge_unreviewed",
            phase_after="report_only_bridge_ready_copy_repair_blocked",
            decision="PASS",
            outcome_status="ADVANCED",
            decision_delta="The reviewed bridge is ready for report-only reads while source-semantic blockers keep canonical copy repair blocked.",
            evidence_refs=[bridge_ref],
            blocks=["canonical_copy_repair"],
            does_not_block=[
                "use_report_only_historical_identity_bridge",
                "offline_historical_analysis",
            ],
            invalidation_conditions=[
                "The proof hash, bridge digest, reviewed artifact set, or reader blockers changes."
            ],
            reopen_conditions=[
                "A new reviewed bridge evidence hash is supplied.",
                "The three source-semantic blockers are repaired and revalidated on a DB copy.",
            ],
        ),
    ]


def manifest_text(entries: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for entry in entries
    )


def write_manifest(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest_text(entries), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor-summary", required=True, type=Path)
    parser.add_argument("--evaluation-results", required=True, type=Path)
    parser.add_argument("--strict-overlap", required=True, type=Path)
    parser.add_argument("--bridge-proof", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    entries = build_seed_entries(
        floor_summary=args.floor_summary,
        evaluation_results=args.evaluation_results,
        strict_overlap=args.strict_overlap,
        bridge_proof=args.bridge_proof,
    )
    output = args.output.resolve(strict=False)
    allowed_root = (ROOT / "docs" / "agent_decisions").resolve()
    if not output.is_relative_to(allowed_root):
        raise SeedEvidenceError(f"output must be inside {allowed_root}")
    expected = manifest_text(entries)
    if args.check:
        observed = output.read_text(encoding="utf-8") if output.is_file() else ""
        if observed != expected:
            raise SeedEvidenceError(f"seed manifest differs from {output}")
        print(f"seed manifest verified: {output} (4 entries)")
        return 0
    write_manifest(output, entries)
    print(f"seed manifest written: {output} (4 entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
