"""Publish a provenance-bound, pre-capture manual scoring readiness index.

This surface is deliberately separate from the autonomous current-race index.
It validates the exact selected race and form/runner source needed to enter one
bounded GHU-051 odds capture, but it does not require or create odds.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from race_collection.synchronous_manual_capture import (
    CaptureOneRejected,
    _atomic_replace_canonical,
    _normalize_current_index_rows,
    _RetainedSafeFiles,
    _v2_runner_rows,
    canonical_bytes,
    sha256_bytes,
)
from src.predictor.manual_independent_capture import parse_canonical_json
from src.predictor.market_form_residual import (
    EFFECTIVE_STATE_SCHEMA,
    ResidualContractError,
    load_frozen_model,
)
from src.predictor.scoring_parity import (
    NUMERIC_CANONICALIZATION_SHA256,
    SCORING_CONFIG_SCHEMA,
    SCORING_CONFIG_SHA256,
)
from utils.csv_metadata import canonical_thedogs_race_identity

READINESS_SCHEMA = "manual_prediction_scoring_readiness_index_v1"
READINESS_FILENAME = "manual_prediction_scoring_readiness_index.json"
READINESS_MAX_BYTES = 2 * 1024 * 1024
MANUAL_CONFIG_RELATIVE = Path(
    "configs/prediction/manual-independent-capture-v1/example-config.json"
)
PREDICTION_CONFIG_RELATIVE = Path("configs/prediction/manual-default.json")
READINESS_SCHEMA_RELATIVE = Path(
    "configs/prediction/manual-readiness-v1/scoring-readiness.schema.json"
)
MODEL_RELATIVE = Path("artifacts/frozen_models/market_form_residual_v1/model.json")
MANIFEST_RELATIVE = Path(
    "artifacts/frozen_models/market_form_residual_v1/manifest.json"
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFETY = {
    "research_only": True,
    "canonical": False,
    "phase7_excluded": True,
    "phase7_eligible": False,
    "phase7_exclusion_reason": "manual_research_only_noncanonical",
}

# These are the accepted lane merge points on the requested exact base.  The
# file hashes below still bind the actual checked-out implementation and
# schemas; the merge points make the ticket lineage explicit in the packet.
GHU_MERGE_COMMITS = {
    "GHU-050": "9a638963ec78c772ec7b19c20961010b37c6cea3",
    "GHU-051": "47e76063cfa14d697a4f4805f75aeaf9d597762e",
    "GHU-052": "492500684fb017b29c3af9748b00e9af8505b457",
    "GHU-053": "b944ef977bbfbb8fb3deaa80823588d2d2a36ecf",
    "GHU-054": "7f52e142faee24911e9b9b51effd5a9631dc55ea",
    "GHU-055": "a594992b1b9a41712376ee63d72d4342ac9a4906",
    "GHU-056": "a6b5a7bcba93716add6ba57c2d501447589e02a5",
}

# Runtime and schema bytes that define the accepted GHU-050--056 lane.  A
# missing or unreadable member is a GLOBAL fault, never a race exclusion.
GHU_PINNED_FILES = {
    "GHU-050": (
        "src/predictor/manual_independent_capture.py",
        "configs/prediction/manual-independent-capture-v1/config.schema.json",
    ),
    "GHU-051": ("src/predictor/manual_independent_capture_executor.py",),
    "GHU-052": (
        "src/predictor/manual_independent_capture_sealer.py",
        "configs/prediction/manual-independent-capture-v1/evidence-bundle.schema.json",
        "configs/prediction/manual-independent-capture-v1/evidence-manifest.schema.json",
    ),
    "GHU-053": (
        "src/predictor/manual_research_scoring.py",
        "src/predictor/market_form_residual.py",
        "src/predictor/scoring_parity.py",
        "configs/prediction/manual-independent-capture-v1/embedded-form.schema.json",
        "configs/prediction/manual-independent-capture-v1/research-prediction.schema.json",
        "configs/prediction/manual-independent-capture-v1/research-prediction-manifest.schema.json",
        "configs/prediction/market-form-residual-v1/scoring-input.schema.json",
        "configs/prediction/market-form-residual-v1/scoring-core-output.schema.json",
    ),
    "GHU-054": (
        "src/predictor/manual_research_cli.py",
        "configs/prediction/manual-independent-capture-v1/manual-research-adapter-response.schema.json",
    ),
    "GHU-055": ("tests/fixtures/manual_independent_capture_child.py",),
    "GHU-056": (
        "src/predictor/manual_research_deployment.py",
        "src/predictor/manual_research_worker.py",
        "ops/systemd/manual-research-api.service.in",
    ),
}
_AUTHORITATIVE_PATHS = frozenset(
    {
        "race_collection/manual_scoring_readiness.py",
        MANUAL_CONFIG_RELATIVE.as_posix(),
        PREDICTION_CONFIG_RELATIVE.as_posix(),
        READINESS_SCHEMA_RELATIVE.as_posix(),
        MODEL_RELATIVE.as_posix(),
        MANIFEST_RELATIVE.as_posix(),
        *(path for paths in GHU_PINNED_FILES.values() for path in paths),
    }
)


class ManualReadinessRejected(RuntimeError):
    """A deterministic GLOBAL readiness rejection."""

    def __init__(self, code: str, **details: Any) -> None:
        super().__init__(code)
        self.code = code
        self.details = details


def manual_scoring_readiness_index_path(state_path: Path) -> Path:
    """Return the fixed path for the separate manual readiness artifact."""

    return Path(state_path).parent / READINESS_FILENAME


def _reject(code: str, **details: Any) -> ManualReadinessRejected:
    return ManualReadinessRejected(code, **details)


def _git_identity(repo_root: Path) -> tuple[str, str]:
    try:
        result = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-C",
                str(repo_root),
                "rev-parse",
                "HEAD",
                "HEAD^{tree}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        tracked = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-C",
                str(repo_root),
                "diff",
                "--name-only",
                "HEAD",
                "--",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        untracked = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-C",
                str(repo_root),
                "ls-files",
                "--others",
                "--exclude-standard",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _reject("GLOBAL_REPOSITORY_IDENTITY_UNAVAILABLE") from exc
    dirty = {
        path
        for output in (tracked.stdout, untracked.stdout)
        for path in output.splitlines()
        if path
    }
    relevant_dirty = sorted(dirty & _AUTHORITATIVE_PATHS)
    if relevant_dirty:
        raise _reject("GLOBAL_REPOSITORY_DIRTY", paths=relevant_dirty)
    values = result.stdout.splitlines()
    if len(values) != 2 or not _GIT_RE.fullmatch(values[0]) or not _GIT_RE.fullmatch(values[1]):
        raise _reject("GLOBAL_REPOSITORY_IDENTITY_INVALID")
    return values[0], values[1]


def _git_tree_bytes(repo_root: Path, source_commit: str, relative: str) -> bytes:
    try:
        result = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-C",
                str(repo_root),
                "show",
                f"{source_commit}:{relative}",
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _reject("GLOBAL_PINNED_IDENTITY_UNAVAILABLE", path=relative) from exc
    return result.stdout


def _repo_bytes(repo_root: Path, relative: str, source_commit: str) -> bytes:
    path = repo_root / relative
    try:
        if path.is_symlink() or not path.is_file() or path.resolve().relative_to(repo_root.resolve()) != Path(relative):
            raise _reject("GLOBAL_PINNED_FILE_UNSAFE", path=relative)
        raw = path.read_bytes()
    except ManualReadinessRejected:
        raise
    except OSError as exc:
        raise _reject("GLOBAL_PINNED_FILE_UNAVAILABLE", path=relative) from exc
    if not raw:
        raise _reject("GLOBAL_PINNED_FILE_EMPTY", path=relative)
    if raw != _git_tree_bytes(repo_root, source_commit, relative):
        raise _reject("GLOBAL_PINNED_IDENTITY_MISMATCH", path=relative)
    return raw


def _sha256(raw: bytes, field: str) -> str:
    value = hashlib.sha256(raw).hexdigest()
    if not _HASH_RE.fullmatch(value):
        raise _reject("GLOBAL_HASH_INVALID", field=field)
    return value


def _pinned_lane_identity(repo_root: Path, source_commit: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for ticket, paths in GHU_PINNED_FILES.items():
        try:
            ancestry = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "merge-base",
                    "--is-ancestor",
                    GHU_MERGE_COMMITS[ticket],
                    source_commit,
                ],
                check=False,
                capture_output=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise _reject("GLOBAL_PINNED_IDENTITY_UNAVAILABLE", ticket=ticket) from exc
        if ancestry.returncode != 0:
            raise _reject("GLOBAL_PINNED_IDENTITY_MISMATCH", ticket=ticket)
        members = []
        for relative in paths:
            raw = _repo_bytes(repo_root, relative, source_commit)
            members.append({"path": relative, "bytes": len(raw), "sha256": _sha256(raw, relative)})
        result[ticket] = {
            "merge_commit": GHU_MERGE_COMMITS[ticket],
            "members": members,
        }
    return result


def _read_pinned_config(
    repo_root: Path, relative: Path, source_commit: str
) -> tuple[Mapping[str, Any], str]:
    raw = _repo_bytes(repo_root, relative.as_posix(), source_commit)
    try:
        value = parse_canonical_json(raw, max_bytes=256 * 1024)
    except Exception as exc:
        raise _reject("GLOBAL_CONFIG_NOT_CANONICAL", path=relative.as_posix()) from exc
    if not isinstance(value, Mapping):
        raise _reject("GLOBAL_CONFIG_INVALID", path=relative.as_posix())
    return value, _sha256(raw, relative.as_posix())


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate:{key}")
        value[key] = item
    return value


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item)


def _model_identity(repo_root: Path, source_commit: str) -> dict[str, Any]:
    model_raw = _repo_bytes(repo_root, MODEL_RELATIVE.as_posix(), source_commit)
    manifest_raw = _repo_bytes(repo_root, MANIFEST_RELATIVE.as_posix(), source_commit)
    try:
        model = load_frozen_model(
            repo_root / MODEL_RELATIVE,
            repo_root / MANIFEST_RELATIVE,
        )
    except (OSError, KeyError, TypeError, ValueError, ResidualContractError) as exc:
        raise _reject("GLOBAL_MODEL_IDENTITY_INVALID") from exc
    return {
        "model_id": "market_form_residual_v1",
        "model_sha256": _sha256(model_raw, "model"),
        "manifest_sha256": _sha256(manifest_raw, "manifest"),
        "effective_state_schema": EFFECTIVE_STATE_SCHEMA,
        "effective_state_sha256": model.effective_state_sha256,
    }


def _source_locator(path: Path, evidence_root: Path) -> str:
    try:
        return path.absolute().relative_to(evidence_root.absolute()).as_posix()
    except ValueError as exc:
        raise _reject("GLOBAL_SOURCE_PATH_UNSAFE") from exc


def _race_identity(race: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    canonical = canonical_thedogs_race_identity(race.get("race_url"))
    if canonical is None or canonical["canonical_url"] != race["race_url"]:
        raise _reject("RACE_IDENTITY_INVALID")
    race_document = {
        "url": race["race_url"],
        "race_id": race["race_id"],
        "race_date": race["date"],
        "venue": race["venue"],
        "venue_slug": canonical["venue_slug"],
        "race_number": race["race_number"],
        "scheduled_start": race["jump_datetime"],
    }
    return race_document, sha256_bytes(canonical_bytes(race_document))


def _runner_set_hash(runner_ids: list[str]) -> str:
    return sha256_bytes(("\n".join(sorted(runner_ids)) + "\n").encode("utf-8"))


def _runner_id(race_id: str, box: int, name: str) -> str:
    token = re.sub(r"[^A-Z0-9]", "", name.upper())
    if not token:
        raise _reject("RUNNER_SET_INVALID")
    return f"{race_id}|box:{box}|dog:{token}"


def _race_source_identity(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"selected_index": index, "race_id": None, "race_url": None}
    return {
        "selected_index": index,
        "race_id": raw.get("race_id") if isinstance(raw.get("race_id"), str) else None,
        "race_url": raw.get("race_url") if isinstance(raw.get("race_url"), str) else None,
    }


def _race_failure(exc: BaseException) -> tuple[str, str | None]:
    if isinstance(exc, ManualReadinessRejected):
        return exc.code, None
    if isinstance(exc, CaptureOneRejected):
        detail = str(exc.details.get("reason") or "")
        if exc.code == "CURRENT_INDEX_PATH_UNSAFE":
            return "GLOBAL_SOURCE_PATH_UNSAFE", detail or None
        if exc.code == "CURRENT_INDEX_SOURCE_MISSING":
            return "FORM_SOURCE_MISSING", detail or None
        if detail in {"race_identity_invalid", "race_id_mismatch_or_duplicate"}:
            return "RACE_IDENTITY_INVALID", detail
        if "runner" in detail or "csv" in detail:
            return "RUNNER_SET_INVALID", detail or None
        if "jump" in detail or "stale" in detail or "observation" in detail:
            return "PREJUMP_TIMING_INVALID", detail or None
        return "FORM_SOURCE_INVALID", detail or None
    if isinstance(exc, (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError)):
        return "FORM_SOURCE_INVALID", type(exc).__name__
    return "FORM_SOURCE_INVALID", type(exc).__name__


def _validate_source(source_raw: bytes, source_path: Path, evidence_root: Path) -> Mapping[str, Any]:
    if not source_raw or len(source_raw) > READINESS_MAX_BYTES:
        raise _reject("GLOBAL_SOURCE_PACKET_NOT_CANONICAL")
    try:
        source = json.loads(
            source_raw,
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
        _reject_nonfinite(source)
    except (ManualReadinessRejected, CaptureOneRejected, KeyError, TypeError, ValueError, OSError) as exc:
        raise _reject("GLOBAL_SOURCE_PACKET_NOT_CANONICAL") from exc
    if not isinstance(source, Mapping):
        raise _reject("GLOBAL_SOURCE_PACKET_INVALID")
    if source.get("status") not in {"SUCCESS", "METADATA_COVERAGE_INCOMPLETE"} or source.get("dry_run") is True:
        raise _reject("GLOBAL_SOURCE_PACKET_UNSAFE_STATUS")
    if not isinstance(source.get("selected_races"), list) or not isinstance(source.get("selected_count"), int) or isinstance(source.get("selected_count"), bool) or source["selected_count"] != len(source["selected_races"]):
        raise _reject("GLOBAL_SOURCE_PACKET_SELECTION_INVALID")
    coverage = source.get("sidecar_metadata_coverage")
    required_coverage_fields = {
        "schema_version",
        "status",
        "selected_race_count",
        "accepted_selected_csv_count",
        "safe_weather_race_count",
        "safe_track_condition_race_count",
        "safe_both_weather_track_race_count",
        "safe_expert_form_race_count",
        "safe_all_weather_track_expert_form_race_count",
        "races",
    }
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("schema_version") != "prejump_sidecar_metadata_coverage_v1"
        or not required_coverage_fields <= set(coverage)
        or not isinstance(coverage.get("races"), list)
        or coverage.get("selected_race_count") != source["selected_count"]
        or len(coverage["races"]) != source["selected_count"]
        or coverage.get("status")
        not in {"READY", "PARTIAL", "DATA_MISSING", "NOT_REQUESTED_NO_SELECTED_RACES"}
        or any(
            not isinstance(coverage.get(field), int)
            or isinstance(coverage.get(field), bool)
            or coverage[field] < 0
            for field in required_coverage_fields - {"schema_version", "status", "races"}
        )
        or any(
            not isinstance(row, Mapping)
            or not {"race_url", "csv_path", "sidecar_path"} <= set(row)
            or not isinstance(row.get("race_url"), (str, type(None)))
            or not isinstance(row.get("csv_path"), (str, type(None)))
            or not isinstance(row.get("sidecar_path"), (str, type(None)))
            for row in coverage["races"]
        )
    ):
        raise _reject("GLOBAL_SOURCE_PACKET_COVERAGE_INVALID")
    try:
        generated = datetime.fromisoformat(str(source["generated_at"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise _reject("GLOBAL_SOURCE_PACKET_TIMESTAMP_INVALID") from exc
    if generated.tzinfo is None or generated.utcoffset() is None:
        raise _reject("GLOBAL_SOURCE_PACKET_TIMESTAMP_INVALID")
    if _source_locator(source_path, evidence_root) == "":
        raise _reject("GLOBAL_SOURCE_PATH_UNSAFE")
    identities: set[str] = set()
    race_urls_by_id: dict[str, str] = {}
    race_ids_by_url: dict[str, str] = {}
    for raw in source["selected_races"]:
        if not isinstance(raw, Mapping):
            continue
        race_id = raw.get("race_id")
        race_url = raw.get("race_url")
        identity = f"{race_id}\x00{race_url}"
        if identity in identities:
            raise _reject("GLOBAL_PACKET_IDENTITY_AMBIGUOUS")
        identities.add(identity)
        if isinstance(race_id, str) and isinstance(race_url, str):
            if race_urls_by_id.get(race_id, race_url) != race_url:
                raise _reject("GLOBAL_PACKET_IDENTITY_AMBIGUOUS")
            if race_ids_by_url.get(race_url, race_id) != race_id:
                raise _reject("GLOBAL_PACKET_IDENTITY_AMBIGUOUS")
            race_urls_by_id[race_id] = race_url
            race_ids_by_url[race_url] = race_id
    return source


def _validate_global_packet_identities(selected: list[Any]) -> None:
    tokens: dict[str, tuple[int, str]] = {}
    urls: dict[str, int] = {}
    for index, raw in enumerate(selected):
        if not isinstance(raw, Mapping):
            continue
        race_id = raw.get("race_id")
        race_url = raw.get("race_url")
        aliases = raw.get("race_id_aliases")
        if (
            not isinstance(race_id, str)
            or not isinstance(race_url, str)
            or not isinstance(aliases, list)
            or any(not isinstance(alias, str) for alias in aliases)
        ):
            continue
        if race_url in urls and urls[race_url] != index:
            raise _reject("GLOBAL_PACKET_IDENTITY_AMBIGUOUS")
        urls[race_url] = index
        for token in [race_id, *aliases]:
            previous = tokens.get(token)
            if previous is not None and previous[0] != index:
                raise _reject("GLOBAL_PACKET_IDENTITY_AMBIGUOUS")
            tokens[token] = (index, race_url)


def _global_identity(repo_root: Path) -> dict[str, Any]:
    source_commit, source_tree = _git_identity(repo_root)
    _repo_bytes(repo_root, READINESS_SCHEMA_RELATIVE.as_posix(), source_commit)
    manual_config, manual_config_sha = _read_pinned_config(
        repo_root, MANUAL_CONFIG_RELATIVE, source_commit
    )
    prediction_config, prediction_config_sha = _read_pinned_config(
        repo_root, PREDICTION_CONFIG_RELATIVE, source_commit
    )
    if (
        manual_config.get("schema_version") != "manual_independent_capture_config_v1"
        or manual_config.get("contract_version") != "manual-independent-capture-v1"
        or manual_config.get("safety") != _SAFETY
        or manual_config.get("attempt_policy") != {
            "max_capture_attempts": 1,
            "max_concurrent_manual_runs": 1,
            "replay_allowed": False,
            "retries_allowed": False,
        }
        or prediction_config.get("schema_version") != "on_demand_prediction_config_v1"
        or prediction_config.get("model") != "market_form_residual_v1"
    ):
        raise _reject("GLOBAL_CONFIG_CONTRACT_MISMATCH")
    timing = manual_config.get("timing")
    if not isinstance(timing, Mapping) or not isinstance(timing.get("minimum_prejump_margin_seconds"), int) or isinstance(timing.get("minimum_prejump_margin_seconds"), bool) or timing["minimum_prejump_margin_seconds"] < 1:
        raise _reject("GLOBAL_CONFIG_CONTRACT_MISMATCH")
    return {
        "repository": {"commit": source_commit, "tree": source_tree},
        "model": _model_identity(repo_root, source_commit),
        "config": {
            "manual_capture": {"path": MANUAL_CONFIG_RELATIVE.as_posix(), "sha256": manual_config_sha},
            "prediction": {"path": PREDICTION_CONFIG_RELATIVE.as_posix(), "sha256": prediction_config_sha},
            "minimum_prejump_margin_seconds": timing["minimum_prejump_margin_seconds"],
        },
        "scoring_contract": {
            "schema_version": SCORING_CONFIG_SCHEMA,
            "sha256": SCORING_CONFIG_SHA256,
            "numeric_canonicalization_sha256": NUMERIC_CANONICALIZATION_SHA256,
        },
        "ghu_050_056": _pinned_lane_identity(repo_root, source_commit),
    }


def _read_prior_readiness_bytes(
    index_path: Path, *, evidence_root: Path, retained: _RetainedSafeFiles
) -> tuple[bool, bytes | None]:
    root = evidence_root.absolute()
    target = index_path.absolute()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise _reject("GLOBAL_SOURCE_PATH_UNSAFE") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise _reject("GLOBAL_SOURCE_PATH_UNSAFE")
    try:
        named = target.lstat()
    except FileNotFoundError:
        return False, None
    except OSError as exc:
        raise _reject("GLOBAL_SOURCE_PATH_UNSAFE") from exc
    if stat.S_ISLNK(named.st_mode) or not stat.S_ISREG(named.st_mode):
        raise _reject("GLOBAL_SOURCE_PATH_UNSAFE")
    try:
        return True, retained.read(
            target, missing_code="GLOBAL_READINESS_PRIOR_UNAVAILABLE"
        )
    except CaptureOneRejected as exc:
        raise _reject("GLOBAL_READINESS_PRIOR_UNAVAILABLE") from exc


def _restore_readiness_bytes(
    parent_fd: int, target_name: str, prior_exists: bool, prior_raw: bytes | None
) -> None:
    if not prior_exists:
        try:
            os.unlink(target_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.fsync(parent_fd)
        return
    if prior_raw is None:
        raise OSError("missing prior readiness bytes")
    temporary = f".{target_name}.{os.getpid()}.{uuid.uuid4().hex}.rollback"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=parent_fd,
    )
    try:
        written = 0
        while written < len(prior_raw):
            written += os.write(descriptor, prior_raw[written:])
        os.fsync(descriptor)
        os.replace(
            temporary,
            target_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
    finally:
        os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass


def _restore_readiness_path(
    index_path: Path,
    *,
    evidence_root: Path,
    prior_exists: bool,
    prior_raw: bytes | None,
) -> None:
    root = evidence_root.absolute()
    target = index_path.absolute()
    relative = target.relative_to(root)
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise OSError("unsafe readiness rollback path")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors = [os.open(root, flags)]
    try:
        for component in relative.parts[:-1]:
            descriptors.append(os.open(component, flags, dir_fd=descriptors[-1]))
        _restore_readiness_bytes(
            descriptors[-1], relative.parts[-1], prior_exists, prior_raw
        )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _global_report_reason(exc: BaseException) -> str:
    if isinstance(exc, ManualReadinessRejected):
        return exc.code
    if isinstance(exc, CaptureOneRejected):
        if exc.code == "CURRENT_INDEX_PATH_UNSAFE":
            return "GLOBAL_SOURCE_PATH_UNSAFE"
        if exc.code == "CURRENT_INDEX_SIZE_INVALID":
            return "GLOBAL_SOURCE_PACKET_INVALID"
        return exc.code
    return type(exc).__name__


def publish_manual_scoring_readiness_index(
    *,
    state_path: Path,
    evidence_root: Path,
    source_refresh_report_path: Path,
    now: datetime | None = None,
    repo_root: Path | None = None,
    max_races: int = 32,
) -> dict[str, Any]:
    """Publish eligible pre-capture races while preserving prior bytes on reject."""

    index_path = manual_scoring_readiness_index_path(state_path)
    report: dict[str, Any] = {
        "schema_version": "manual_prediction_scoring_readiness_publish_v1",
        "status": "REJECTED",
        "index_path": str(index_path),
        "source_refresh_report_path": str(source_refresh_report_path),
    }
    try:
        if isinstance(max_races, bool) or not isinstance(max_races, int) or not 1 <= max_races <= 256:
            raise _reject("GLOBAL_SELECTION_LIMIT_INVALID")
        generated_at = now or datetime.now().astimezone()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise _reject("GLOBAL_GENERATED_TIMESTAMP_INVALID")
        root = Path(evidence_root).absolute()
        source_path = Path(source_refresh_report_path).absolute()
        identities = _global_identity(Path(repo_root or Path(__file__).resolve().parents[1]).absolute())
        with _RetainedSafeFiles(root) as retained:
            prior_exists, prior_raw = _read_prior_readiness_bytes(
                index_path, evidence_root=root, retained=retained
            )
            source_raw = retained.read(source_path, missing_code="GLOBAL_SOURCE_PACKET_UNAVAILABLE")
            source = _validate_source(source_raw, source_path, root)
            selected = source["selected_races"]
            if len(selected) > max_races:
                raise _reject("GLOBAL_SELECTION_LIMIT_EXCEEDED")
            _validate_global_packet_identities(selected)
            minimum_margin = int(identities["config"]["minimum_prejump_margin_seconds"])
            eligible: list[dict[str, Any]] = []
            exclusions: list[dict[str, Any]] = []
            seen_races: set[str] = set()
            for index, raw in enumerate(selected):
                source_identity = _race_source_identity(raw, index)
                try:
                    one_source = dict(source)
                    one_source["selected_races"] = [raw]
                    one_source["selected_count"] = 1
                    race = _normalize_current_index_rows(one_source, max_races=1)[0]
                    race_document, race_identity_sha = _race_identity(race)
                    jump = datetime.fromisoformat(race["jump_datetime"])
                    if jump <= generated_at or (jump - generated_at).total_seconds() < minimum_margin:
                        raise _reject("PREJUMP_TIMING_INVALID")
                    coverage_rows = source["sidecar_metadata_coverage"]["races"]
                    matching_coverage = [
                        item
                        for item in coverage_rows
                        if isinstance(item, Mapping)
                        and item.get("race_url") == race["race_url"]
                    ]
                    if not matching_coverage:
                        raise _reject("FORM_SOURCE_MISSING")
                    if len(matching_coverage) != 1:
                        raise _reject("FORM_SOURCE_AMBIGUOUS")
                    if not matching_coverage[0].get("csv_path") or not matching_coverage[0].get("sidecar_path"):
                        raise _reject("FORM_SOURCE_MISSING")
                    runners, runner_source, _ = _v2_runner_rows(
                        race, source, evidence_root=root, snapshot=retained
                    )
                    runner_documents = [
                        {
                            "runner_id": _runner_id(race["race_id"], row["box"], row["display_name"]),
                            "box_number": row["box"],
                            "dog_name": row["display_name"],
                            "source_native_runner_id": row["source_native_runner_id"],
                        }
                        for row in runners
                    ]
                    runner_ids = [row["runner_id"] for row in runner_documents]
                    if len(runner_ids) < 2 or len(set(runner_ids)) != len(runner_ids):
                        raise _reject("RUNNER_SET_INVALID")
                    if race["race_id"] in seen_races:
                        raise _reject("GLOBAL_PACKET_IDENTITY_AMBIGUOUS")
                    seen_races.add(race["race_id"])
                    eligible.append(
                        {
                            **race_document,
                            "race_identity_sha256": race_identity_sha,
                            "runner_set_sha256": _runner_set_hash(runner_ids),
                            "runners": runner_documents,
                            "form_source": runner_source,
                            "prejump": {
                                "source_generated_at": runner_source["source_generated_at"],
                                "observed_at": runner_source["observed_at"],
                                "minimum_margin_seconds": minimum_margin,
                            },
                            "odds": {
                                "status": "PENDING_GHU_051",
                                "next_authorized_step": "GHU-051_ONE_RACE_BOUNDED_CAPTURE",
                            },
                        }
                    )
                except (
                    AttributeError,
                    CaptureOneRejected,
                    IndexError,
                    KeyError,
                    ManualReadinessRejected,
                    OSError,
                    TypeError,
                    ValueError,
                ) as exc:
                    if isinstance(exc, CaptureOneRejected) and exc.code == "CURRENT_INDEX_PATH_UNSAFE":
                        raise _reject("GLOBAL_SOURCE_PATH_UNSAFE") from exc
                    code, detail = _race_failure(exc)
                    if code.startswith("GLOBAL_"):
                        raise
                    exclusions.append(
                        {
                            **source_identity,
                            "reason_code": code,
                            "detail": detail,
                        }
                    )
            packet = {
                "schema_version": READINESS_SCHEMA,
                "safety": dict(_SAFETY),
                "generated_at": generated_at.isoformat(),
                "source_refresh_report": {
                    "path": _source_locator(source_path, root),
                    "bytes": len(source_raw),
                    "sha256": sha256_bytes(source_raw),
                    "generated_at": source["generated_at"],
                    "status": source["status"],
                },
                **identities,
                "selected_race_count": len(selected),
                "eligible_race_count": len(eligible),
                "excluded_race_count": len(exclusions),
                "races": eligible,
                "exclusions": exclusions,
            }
            try:
                _atomic_replace_canonical(
                    index_path,
                    packet,
                    evidence_root=root,
                    _on_replace_failure=lambda parent_fd, target_name: _restore_readiness_bytes(
                        parent_fd, target_name, prior_exists, prior_raw
                    ),
                )
            except (CaptureOneRejected, OSError, TypeError, ValueError):
                try:
                    _restore_readiness_path(
                        index_path,
                        evidence_root=root,
                        prior_exists=prior_exists,
                        prior_raw=prior_raw,
                    )
                except OSError as rollback_error:
                    del rollback_error
                raise
            packet_raw = canonical_bytes(packet)
    except (
        AttributeError,
        CaptureOneRejected,
        IndexError,
        KeyError,
        ManualReadinessRejected,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        report["reason"] = _global_report_reason(exc)
        return report
    report.update(
        {
            "status": "PUBLISHED",
            "packet_schema_version": READINESS_SCHEMA,
            "packet_sha256": sha256_bytes(packet_raw),
            "source_refresh_report_sha256": packet["source_refresh_report"]["sha256"],
            "selected_race_count": packet["selected_race_count"],
            "eligible_race_count": packet["eligible_race_count"],
            "excluded_race_count": packet["excluded_race_count"],
        }
    )
    return report


__all__ = [
    "GHU_MERGE_COMMITS",
    "READINESS_FILENAME",
    "READINESS_SCHEMA",
    "ManualReadinessRejected",
    "manual_scoring_readiness_index_path",
    "publish_manual_scoring_readiness_index",
]
