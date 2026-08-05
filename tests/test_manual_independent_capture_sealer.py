from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from src.predictor.manual_independent_capture import (
    PROTECTED_PATH_KEYS,
    ManualIndependentCaptureRejected,
    canonical_bytes,
    canonical_sha256,
)
from src.predictor.manual_independent_capture_executor import (
    FixtureChildLaunch,
    execute_manual_capture_fixture,
)
from src.predictor.manual_independent_capture_sealer import (
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    EVIDENCE_MANIFEST_SCHEMA_VERSION,
    SEALED_ROOT_NAME,
    ManualEvidenceRejected,
    build_sealing_identity,
    expectations_from_execution,
    seal_manual_capture,
    verify_manual_evidence_bundle,
)
from src.predictor.on_demand import sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "tests/fixtures/manual_independent_capture_child.py"
SOURCE_COMMIT = "47e76063cfa14d697a4f4805f75aeaf9d597762e"
SOURCE_TREE = "5cc7625500e0d84979de365e5155b45ef28df6af"
MODEL_BYTES = b'{"model":"fixture-only-no-scoring"}\n'
NOW = datetime(2026, 8, 5, 1, 0, 0, tzinfo=timezone.utc)


def _race(*, scheduled: datetime | None = None) -> dict:
    scheduled = scheduled or NOW + timedelta(hours=1)
    return {
        "url": "https://www.thedogs.com.au/racing/richmond/2026-08-05/1/race-name",
        "race_id": "Race 1 - RICH - 2026-08-05",
        "race_date": "2026-08-05",
        "venue": "RICH",
        "venue_slug": "richmond",
        "race_number": 1,
        "scheduled_start": scheduled.isoformat(),
    }


def _config(tmp_path: Path, **timing: int) -> dict:
    operations_root = tmp_path / "manual-operations"
    operations_root.mkdir()
    manual_root = operations_root / "manual-independent-capture-v1"
    return {
        "schema_version": "manual_independent_capture_config_v1",
        "contract_version": "manual-independent-capture-v1",
        "safety": {
            "research_only": True,
            "canonical": False,
            "phase7_excluded": True,
            "phase7_eligible": False,
            "phase7_exclusion_reason": "manual_research_only_noncanonical",
        },
        "authority_profile": "manual_independent_capture_research_only_v1",
        "paths": {
            "operations_root": str(operations_root),
            "manual_root": str(manual_root),
            "browser_profile": str(manual_root / "browser-profile"),
            "runs_root": str(manual_root / "runs"),
            "manual_lock": str(manual_root / "manual-capture.lock"),
        },
        "timing": {
            "minimum_prejump_margin_seconds": timing.get("minimum", 120),
            "hard_timeout_seconds": timing.get("timeout", 5),
            "cancellation_grace_seconds": timing.get("grace", 1),
        },
        "attempt_policy": {
            "max_concurrent_manual_runs": 1,
            "max_capture_attempts": 1,
            "retries_allowed": False,
            "replay_allowed": False,
        },
    }


def _forbidden(tmp_path: Path, *, sentinel: Path | None = None) -> dict[str, str]:
    protected = tmp_path / "protected"
    protected.mkdir(exist_ok=True)
    values = {
        name: str(protected / name.replace("_", "-"))
        for name in PROTECTED_PATH_KEYS
    }
    if sentinel is not None:
        values["autonomous_shared_lock"] = str(sentinel)
    return values


def _command(mode: str, source_timestamp: str = NOW.isoformat()):
    def build(launch: FixtureChildLaunch) -> list[str]:
        return [
            sys.executable,
            str(CHILD),
            mode,
            "--source-timestamp",
            source_timestamp,
            "--race-sha",
            canonical_sha256(launch.selected_race),
        ]

    return build


def _execute(
    tmp_path: Path,
    *,
    cfg: dict,
    forbidden: dict[str, str],
    race: dict | None = None,
    mode: str = "success",
    cancellation_token=None,
):
    selected = race or _race()
    return execute_manual_capture_fixture(
        config=cfg,
        forbidden_paths=forbidden,
        requested_race_url=selected["url"],
        selected_race=selected,
        model_bytes=MODEL_BYTES,
        source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE,
        fixture_child_command=_command(mode),
        cancellation_token=cancellation_token,
        now=lambda: NOW,
    )


def _context(tmp_path: Path, *, mode: str = "success", race: dict | None = None):
    cfg = _config(tmp_path)
    forbidden = _forbidden(tmp_path)
    execution = _execute(
        tmp_path, cfg=cfg, forbidden=forbidden, mode=mode, race=race
    )
    expected = expectations_from_execution(execution)
    identity = build_sealing_identity(
        repo_root=ROOT,
        source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE,
    )
    return cfg, forbidden, execution, expected, identity


def _seal(tmp_path: Path, **kwargs):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)
    result = seal_manual_capture(
        execution,
        config=cfg,
        forbidden_paths=forbidden,
        expected=expected,
        identity=identity,
        repo_root=ROOT,
        **kwargs,
    )
    return cfg, forbidden, execution, expected, identity, result


def _make_writable(path: Path) -> None:
    for current, directories, filenames in os.walk(path):
        os.chmod(current, 0o700)
        for name in directories:
            os.chmod(Path(current) / name, 0o700)
        for name in filenames:
            os.chmod(Path(current) / name, 0o600)


def _rewrite_member_and_manifest(bundle_dir: Path, relative: str, value: dict) -> None:
    _make_writable(bundle_dir)
    target = bundle_dir / relative
    raw = canonical_bytes(value)
    target.write_bytes(raw)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for member in manifest["members"]:
        if member["path"] == relative:
            member["bytes"] = len(raw)
            member["sha256"] = sha256_bytes(raw)
    manifest_path.write_bytes(canonical_bytes(manifest))


def test_valid_fixture_capture_seals_and_read_only_verifier_accepts(tmp_path: Path):
    cfg, forbidden, execution, expected, identity, result = _seal(tmp_path)

    assert result.replayed is False
    assert result.bundle["schema_version"] == EVIDENCE_BUNDLE_SCHEMA_VERSION
    assert result.manifest["schema_version"] == EVIDENCE_MANIFEST_SCHEMA_VERSION
    assert result.bundle["safety"]["research_only"] is True
    assert result.bundle["safety"]["canonical"] is False
    assert result.bundle["safety"]["phase7_excluded"] is True
    assert result.bundle["closure"]["outcome_accessed"] is False
    assert result.bundle["closure"]["canonical_accessed"] is False
    assert result.bundle["closure"]["phase7_accessed"] is False
    assert result.bundle["attempt"] == {
        "attempt_count": 1,
        "source_attempt_count": 1,
    }
    assert result.bundle["source"]["final_url"] == _race()["url"]
    assert result.bundle["source"]["status_code"] == 200
    assert result.bundle["source"]["content_type"] == "text/csv; charset=utf-8"
    assert result.bundle["cleanup"]["confirmed"] is True
    assert result.bundle["timing"]["final_prejump_margin_seconds"] == 3600
    assert verify_manual_evidence_bundle(
        result.bundle_dir,
        run_dir=execution.run_dir,
        expected=expected,
        expected_identity=identity,
    ).bundle == result.bundle
    assert cfg["paths"]["manual_root"] in str(result.bundle_dir)
    assert forbidden["canonical_database"] not in str(result.bundle_dir)


def test_bundle_layout_preserves_raw_source_and_normalizes_odds_separately(
    tmp_path: Path,
):
    _, _, execution, _, _, result = _seal(tmp_path)
    files = {
        path.relative_to(result.bundle_dir).as_posix()
        for path in result.bundle_dir.rglob("*")
        if path.is_file()
    }
    assert files == {
        "bundle.json",
        "manifest.json",
        "normalized/odds.json",
        "producer/terminal.json",
        "source/raw.bin",
    }
    assert result.bundle_dir.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0 for path in result.bundle_dir.rglob("*"))
    source_path = execution.artifact["provenance"]["source_files"][0]["path"]
    assert (result.bundle_dir / "source/raw.bin").read_bytes() == (
        execution.run_dir / source_path
    ).read_bytes()
    odds = json.loads((result.bundle_dir / "normalized/odds.json").read_bytes())
    assert [row["box_number"] for row in odds["runners"]] == [1, 2]
    assert [row["decimal_odds"] for row in odds["runners"]] == [2.5, 3.75]
    assert result.bundle["source"]["odds_parser"] == "manual_fixture_csv_odds_v1"
    assert result.bundle["source"]["parsed_odds_sha256"] == canonical_sha256(
        [
            {"box_number": 1, "display_name": "Alpha Dog", "decimal_odds": 2.5},
            {"box_number": 2, "display_name": "Beta Dog", "decimal_odds": 3.75},
        ]
    )


def test_bundle_and_manifest_conform_to_versioned_schemas(tmp_path: Path):
    *_, result = _seal(tmp_path)
    for schema_name, value in (
        ("evidence-bundle.schema.json", result.bundle),
        ("evidence-manifest.schema.json", result.manifest),
    ):
        schema = json.loads(
            (
                ROOT
                / "configs/prediction/manual-independent-capture-v1"
                / schema_name
            ).read_bytes()
        )
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(
            schema, format_checker=FormatChecker()
        ).validate(value)
        assert schema["additionalProperties"] is False
        assert set(value) == set(schema["required"])
        assert schema["properties"]["schema_version"]["const"] == value[
            "schema_version"
        ]
        assert schema["properties"]["contract_version"]["const"] == value[
            "contract_version"
        ]
        safety = schema["$defs"]["safety"]["properties"]
        assert safety["research_only"] == {"const": True}
        assert safety["canonical"] == {"const": False}
        assert safety["phase7_excluded"] == {"const": True}


def test_exact_replay_returns_identical_bundle_without_republication(tmp_path: Path):
    cfg, forbidden, execution, expected, identity, first = _seal(tmp_path)
    before = {
        path.relative_to(first.bundle_dir).as_posix(): path.read_bytes()
        for path in first.bundle_dir.rglob("*")
        if path.is_file()
    }

    second = seal_manual_capture(
        execution,
        config=cfg,
        forbidden_paths=forbidden,
        expected=expected,
        identity=identity,
        repo_root=ROOT,
    )

    assert second.replayed is True
    assert second.bundle_dir == first.bundle_dir
    assert second.manifest_sha256 == first.manifest_sha256
    assert before == {
        path.relative_to(second.bundle_dir).as_posix(): path.read_bytes()
        for path in second.bundle_dir.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("source_commit", "0" * 40),
        ("source_tree", "0" * 40),
        ("model_sha256", "0" * 64),
        ("config_sha256", "0" * 64),
        ("run_id", "00000000-0000-4000-8000-000000000000"),
        ("request_id", "00000000-0000-4000-8000-000000000000"),
        ("request_sha256", "0" * 64),
        ("race_identity_sha256", "0" * 64),
        ("runner_set_sha256", "0" * 64),
        ("odds_sha256", "0" * 64),
        ("source_sha256", "0" * 64),
        ("source_timestamp", "2026-08-05T01:00:01+00:00"),
        (
            "final_url",
            "https://www.thedogs.com.au/racing/richmond/2026-08-05/2/other",
        ),
        ("status_code", 201),
        ("content_type", "application/json"),
        ("terminal_sha256", "0" * 64),
        ("cleanup_sha256", "0" * 64),
    ),
)
def test_expected_identity_runner_odds_config_and_hash_disagreement_fails_closed(
    tmp_path: Path, field: str, replacement
):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)

    with pytest.raises((ManualEvidenceRejected, ManualIndependentCaptureRejected)):
        seal_manual_capture(
            execution,
            config=cfg,
            forbidden_paths=forbidden,
            expected=replace(expected, **{field: replacement}),
            identity=identity,
            repo_root=ROOT,
        )


def test_config_disagreement_fails_closed(tmp_path: Path):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)
    changed = deepcopy(cfg)
    changed["timing"]["minimum_prejump_margin_seconds"] += 1

    with pytest.raises((ManualEvidenceRejected, ManualIndependentCaptureRejected)):
        seal_manual_capture(
            execution,
            config=changed,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("final_url", "https://www.thedogs.com.au/racing/richmond/2026-08-05/2/other"),
        ("status_code", 302),
        ("content_type", "application/json"),
        ("content_type", "text/csv\nresult"),
        ("body_sha256", "0" * 64),
    ),
)
def test_source_response_url_status_content_type_and_hash_mismatch_rejected(
    tmp_path: Path, field: str, value
):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)
    changed = replace(
        execution,
        source_response=replace(execution.source_response, **{field: value}),
    )

    with pytest.raises(ManualEvidenceRejected):
        seal_manual_capture(
            changed,
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
        )


def test_odds_must_be_derived_from_preserved_source_bytes(tmp_path: Path):
    cfg, forbidden, execution, _, identity = _context(tmp_path)
    raw = b"box,dog,decimal_odds\n1,Alpha Dog,2.6\n2,Beta Dog,3.75\n"
    artifact = deepcopy(execution.artifact)
    source_row = artifact["provenance"]["source_files"][0]
    source_path = execution.run_dir / source_row["path"]
    source_path.write_bytes(raw)
    source_row["bytes"] = len(raw)
    source_row["sha256"] = sha256_bytes(raw)
    execution.terminal_path.write_bytes(canonical_bytes(artifact))
    changed = replace(
        execution,
        artifact=artifact,
        source_response=replace(
            execution.source_response, body_sha256=sha256_bytes(raw)
        ),
    )
    expected = expectations_from_execution(changed)

    with pytest.raises(ManualEvidenceRejected, match="ODDS_PROVENANCE_AMBIGUOUS"):
        seal_manual_capture(
            changed,
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
        )


def test_late_capture_is_never_sealable(tmp_path: Path):
    cfg = _config(tmp_path)
    forbidden = _forbidden(tmp_path)
    execution = _execute(
        tmp_path,
        cfg=cfg,
        forbidden=forbidden,
        race=_race(scheduled=NOW + timedelta(seconds=119)),
    )
    expected = expectations_from_execution(execution)
    identity = build_sealing_identity(
        repo_root=ROOT, source_commit=SOURCE_COMMIT, source_tree=SOURCE_TREE
    )

    with pytest.raises(ManualEvidenceRejected, match="CAPTURE_NOT_SEALABLE"):
        seal_manual_capture(
            execution,
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
        )


def test_cancelled_capture_and_unconfirmed_cleanup_are_never_sealable(tmp_path: Path):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)
    unconfirmed = replace(execution.cleanup, confirmed=False)
    with pytest.raises(ManualEvidenceRejected, match="CAPTURE_NOT_SEALABLE"):
        seal_manual_capture(
            replace(execution, cleanup=unconfirmed),
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
        )
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(ManualEvidenceRejected, match="CANCELLED"):
        seal_manual_capture(
            execution,
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
            cancellation_token=cancelled,
        )


@pytest.mark.parametrize("mode", ("outcome-json", "outcome-csv", "outcome-html"))
def test_outcome_material_in_source_is_rejected(tmp_path: Path, mode: str):
    cfg, forbidden, execution, expected, identity = _context(tmp_path, mode=mode)

    with pytest.raises(ManualEvidenceRejected, match="OUTCOME_MATERIAL_FORBIDDEN"):
        seal_manual_capture(
            execution,
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
        )


@pytest.mark.parametrize(
    "stage",
    (
        "stage_created",
        "member_written:bundle.json",
        "member_written:normalized/odds.json",
        "member_written:producer/terminal.json",
        "member_written:source/raw.bin",
        "manifest_written",
        "members_written",
        "stage_fsynced",
        "renamed",
        "parent_fsynced",
    ),
)
def test_interruption_at_every_publication_stage_never_exposes_partial_bundle(
    tmp_path: Path, stage: str
):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)

    def interrupt(name: str, _path: Path) -> None:
        if name == stage:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        seal_manual_capture(
            execution,
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
            stage_hook=interrupt,
        )
    sealed_root = execution.run_dir / SEALED_ROOT_NAME
    visible = [path for path in sealed_root.iterdir() if not path.name.startswith(".")]
    if stage not in {"renamed", "parent_fsynced"}:
        assert visible == []
    else:
        assert len(visible) == 1
        verify_manual_evidence_bundle(
            visible[0],
            run_dir=execution.run_dir,
            expected=expected,
            expected_identity=identity,
        )
    replay = seal_manual_capture(
        execution,
        config=cfg,
        forbidden_paths=forbidden,
        expected=expected,
        identity=identity,
        repo_root=ROOT,
    )
    assert not list(sealed_root.glob(".tmp-evidence-*"))
    assert replay.bundle_dir.is_dir()


@pytest.mark.parametrize(
    "stage", ("stage_created", "member_written:bundle.json", "stage_fsynced")
)
def test_cancellation_during_publication_cannot_publish(tmp_path: Path, stage: str):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)
    cancelled = threading.Event()

    def cancel(name: str, _path: Path) -> None:
        if name == stage:
            cancelled.set()

    with pytest.raises(ManualEvidenceRejected, match="CANCELLED"):
        seal_manual_capture(
            execution,
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
            cancellation_token=cancelled,
            stage_hook=cancel,
        )
    sealed_root = execution.run_dir / SEALED_ROOT_NAME
    assert [path for path in sealed_root.iterdir() if not path.name.startswith(".")] == []


def test_stale_temp_recovery_is_bounded_to_plain_stage_directories(tmp_path: Path):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)
    sealed_root = execution.run_dir / SEALED_ROOT_NAME
    sealed_root.mkdir()
    stale = sealed_root / ".tmp-evidence-stale"
    (stale / "nested").mkdir(parents=True)
    (stale / "nested/partial").write_bytes(b"partial")

    result = seal_manual_capture(
        execution,
        config=cfg,
        forbidden_paths=forbidden,
        expected=expected,
        identity=identity,
        repo_root=ROOT,
    )

    assert result.bundle_dir.is_dir()
    assert not stale.exists()


def test_stale_temp_symlink_fails_closed_without_touching_target(tmp_path: Path):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)
    sealed_root = execution.run_dir / SEALED_ROOT_NAME
    sealed_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"preserve")
    (sealed_root / ".tmp-evidence-hostile").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ManualEvidenceRejected, match="UNSAFE_STALE_TEMP"):
        seal_manual_capture(
            execution,
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
        )
    assert sentinel.read_bytes() == b"preserve"


def test_stale_temp_hardlink_fails_closed_without_chmod_or_unlink(tmp_path: Path):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)
    sealed_root = execution.run_dir / SEALED_ROOT_NAME
    sealed_root.mkdir()
    stale = sealed_root / ".tmp-evidence-hostile"
    stale.mkdir()
    outside = tmp_path / "outside-file"
    outside.write_bytes(b"preserve")
    before_mode = outside.stat().st_mode
    os.link(outside, stale / "hardlink")

    with pytest.raises(ManualEvidenceRejected, match="UNSAFE_STALE_TEMP"):
        seal_manual_capture(
            execution,
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
        )
    assert outside.read_bytes() == b"preserve"
    assert outside.stat().st_mode == before_mode
    assert (stale / "hardlink").exists()


def test_concurrent_exact_seals_publish_once_and_replay_identically(tmp_path: Path):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)

    def run():
        return seal_manual_capture(
            execution,
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: run(), range(2)))

    assert results[0].bundle_dir == results[1].bundle_dir
    assert results[0].manifest_sha256 == results[1].manifest_sha256
    assert sorted(result.replayed for result in results) == [False, True]
    sealed_root = execution.run_dir / SEALED_ROOT_NAME
    assert len([path for path in sealed_root.iterdir() if not path.name.startswith(".")]) == 1


def test_run_root_escape_and_source_symlink_are_rejected(tmp_path: Path):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)
    escaped = tmp_path / execution.run_dir.name
    escaped.mkdir()
    with pytest.raises(ManualEvidenceRejected, match="UNSAFE_PATH"):
        seal_manual_capture(
            replace(execution, run_dir=escaped, terminal_path=escaped / "terminal.json"),
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
        )
    source = execution.run_dir / execution.artifact["provenance"]["source_files"][0][
        "path"
    ]
    outside = tmp_path / "outside-source"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(ManualEvidenceRejected, match="UNSAFE_PATH"):
        seal_manual_capture(
            execution,
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
        )


@pytest.mark.parametrize("unsafe_path", ("../outside.csv", "/etc/passwd"))
def test_forged_producer_member_paths_are_rejected_before_open(
    tmp_path: Path, unsafe_path: str
):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)
    artifact = deepcopy(execution.artifact)
    artifact["provenance"]["source_files"][0]["path"] = unsafe_path
    execution.terminal_path.write_bytes(canonical_bytes(artifact))
    changed = replace(execution, artifact=artifact)
    expected = expectations_from_execution(changed)

    with pytest.raises(ManualEvidenceRejected, match="UNSAFE_PATH"):
        seal_manual_capture(
            changed,
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
        )


def test_intermediate_producer_directory_symlink_is_rejected(tmp_path: Path):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)
    sources = execution.run_dir / "sources"
    outside = tmp_path / "outside-sources"
    sources.rename(outside)
    sources.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ManualEvidenceRejected, match="UNSAFE_PATH"):
        seal_manual_capture(
            execution,
            config=cfg,
            forbidden_paths=forbidden,
            expected=expected,
            identity=identity,
            repo_root=ROOT,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("pid", 0), ("reason", "clean\nup"), ("term_sent", "false")),
)
def test_cleanup_schema_and_runtime_reject_the_same_invalid_values(
    tmp_path: Path, field: str, value
):
    _, _, execution, expected, identity, result = _seal(tmp_path)
    schema = json.loads(
        (
            ROOT
            / "configs/prediction/manual-independent-capture-v1"
            / "evidence-bundle.schema.json"
        ).read_bytes()
    )
    bundle = deepcopy(result.bundle)
    bundle["cleanup"][field] = value
    with pytest.raises(ValidationError):
        Draft202012Validator(
            schema, format_checker=FormatChecker()
        ).validate(bundle)
    _rewrite_member_and_manifest(result.bundle_dir, "bundle.json", bundle)
    with pytest.raises(ManualEvidenceRejected):
        verify_manual_evidence_bundle(
            result.bundle_dir,
            run_dir=execution.run_dir,
            expected=expected,
            expected_identity=identity,
        )


def test_content_type_schema_and_runtime_reject_non_ascii(tmp_path: Path):
    _, _, execution, expected, identity, result = _seal(tmp_path)
    schema = json.loads(
        (
            ROOT
            / "configs/prediction/manual-independent-capture-v1"
            / "evidence-bundle.schema.json"
        ).read_bytes()
    )
    bundle = deepcopy(result.bundle)
    bundle["source"]["content_type"] = "text/csv; charset=utf-8é"
    with pytest.raises(ValidationError):
        Draft202012Validator(
            schema, format_checker=FormatChecker()
        ).validate(bundle)
    _rewrite_member_and_manifest(result.bundle_dir, "bundle.json", bundle)
    with pytest.raises(ManualEvidenceRejected):
        verify_manual_evidence_bundle(
            result.bundle_dir,
            run_dir=execution.run_dir,
            expected=expected,
            expected_identity=identity,
        )


def test_attacker_rehashed_minimum_margin_tampering_is_rejected(tmp_path: Path):
    _, _, execution, expected, identity, result = _seal(tmp_path)
    bundle = deepcopy(result.bundle)
    bundle["timing"]["minimum_prejump_margin_seconds"] = 1
    _rewrite_member_and_manifest(result.bundle_dir, "bundle.json", bundle)

    with pytest.raises(ManualEvidenceRejected, match="PRODUCER_PROVENANCE_MISMATCH"):
        verify_manual_evidence_bundle(
            result.bundle_dir,
            run_dir=execution.run_dir,
            expected=expected,
            expected_identity=identity,
        )


def test_reader_rejects_partial_temp_and_final_directories(tmp_path: Path):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)
    sealed_root = execution.run_dir / SEALED_ROOT_NAME
    sealed_root.mkdir()
    temp = sealed_root / ".tmp-evidence-dead"
    temp.mkdir()
    partial = sealed_root / ("0" * 64)
    partial.mkdir()
    with pytest.raises(ManualEvidenceRejected):
        verify_manual_evidence_bundle(
            temp,
            run_dir=execution.run_dir,
            expected=expected,
            expected_identity=identity,
        )
    with pytest.raises(ManualEvidenceRejected):
        verify_manual_evidence_bundle(
            partial,
            run_dir=execution.run_dir,
            expected=expected,
            expected_identity=identity,
        )
    assert cfg and forbidden


def test_post_seal_tampering_is_detected_even_when_attacker_rehashes_manifest(
    tmp_path: Path,
):
    _, _, execution, expected, identity, result = _seal(tmp_path)
    _make_writable(result.bundle_dir)
    raw_path = result.bundle_dir / "source/raw.bin"
    raw_path.write_bytes(raw_path.read_bytes() + b"tamper")
    with pytest.raises(ManualEvidenceRejected, match="HASH_DRIFT"):
        verify_manual_evidence_bundle(
            result.bundle_dir,
            run_dir=execution.run_dir,
            expected=expected,
            expected_identity=identity,
        )

    producer_source = execution.run_dir / execution.artifact["provenance"][
        "source_files"
    ][0]["path"]
    raw_path.write_bytes(producer_source.read_bytes())
    (result.bundle_dir / "manifest.json").write_bytes(canonical_bytes(result.manifest))
    bundle = deepcopy(result.bundle)
    bundle["outcome"] = {"winner": 1}
    _rewrite_member_and_manifest(result.bundle_dir, "bundle.json", bundle)
    with pytest.raises(ManualEvidenceRejected):
        verify_manual_evidence_bundle(
            result.bundle_dir,
            run_dir=execution.run_dir,
            expected=expected,
            expected_identity=identity,
        )


def test_autonomous_and_canonical_sentinels_remain_byte_and_metadata_unchanged(
    tmp_path: Path,
):
    sentinel = tmp_path / "autonomous.lock"
    sentinel.write_bytes(b"AUTONOMOUS-SENTINEL")
    before = (sentinel.read_bytes(), sentinel.stat().st_mtime_ns, sentinel.stat().st_mode)
    cfg = _config(tmp_path)
    forbidden = _forbidden(tmp_path, sentinel=sentinel)
    execution = _execute(tmp_path, cfg=cfg, forbidden=forbidden)
    expected = expectations_from_execution(execution)
    identity = build_sealing_identity(
        repo_root=ROOT, source_commit=SOURCE_COMMIT, source_tree=SOURCE_TREE
    )

    seal_manual_capture(
        execution,
        config=cfg,
        forbidden_paths=forbidden,
        expected=expected,
        identity=identity,
        repo_root=ROOT,
    )

    assert (sentinel.read_bytes(), sentinel.stat().st_mtime_ns, sentinel.stat().st_mode) == before


def test_seal_and_verify_never_open_sqlite(tmp_path: Path, monkeypatch):
    cfg, forbidden, execution, expected, identity = _context(tmp_path)

    def forbidden_connect(*_args, **_kwargs):
        raise AssertionError("SQLite access is outside GHU-052 authority")

    monkeypatch.setattr(sqlite3, "connect", forbidden_connect)
    result = seal_manual_capture(
        execution,
        config=cfg,
        forbidden_paths=forbidden,
        expected=expected,
        identity=identity,
        repo_root=ROOT,
    )
    verify_manual_evidence_bundle(
        result.bundle_dir,
        run_dir=execution.run_dir,
        expected=expected,
        expected_identity=identity,
    )
