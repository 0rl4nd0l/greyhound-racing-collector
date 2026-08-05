"""Bounded manual CLI/API adapter for the accepted GHU-053 scorer.

The adapter only loads caller-supplied regular files and directories, converts
caller-supplied identity documents to GHU-052/GHU-053 contract objects, and
delegates validation, scoring, publication, and verification to those layers.
It has no capture, discovery, network, database, result, or runtime surface.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

from src.predictor.manual_independent_capture import (
    canonical_bytes,
    parse_canonical_json,
)
from src.predictor.manual_independent_capture_sealer import (
    ManualEvidenceRejected,
    SealExpectations,
    SealingIdentity,
)
from src.predictor.manual_research_scoring import (
    ManualResearchScoringRejected,
    ResearchScoringIdentity,
    score_verified_manual_evidence,
)
from src.predictor.market_form_residual import ResidualContractError, load_frozen_model

RESPONSE_SCHEMA = "manual_research_invocation_response_v1"
VERIFICATION_STATUS = "VERIFIED"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_FORM_BYTES = 4 * 1024 * 1024
_MAX_CONFIG_BYTES = 64 * 1024
_MAX_MODEL_BYTES = 16 * 1024 * 1024
_MAX_METADATA_BYTES = 256 * 1024
_FORBIDDEN_ERROR_TOKENS = frozenset(("EV", "STAKE", "BET", "OUTCOME", "RESULT", "PHASE7"))


class ManualResearchAdapterRejected(RuntimeError):
    """A deterministic fail-closed adapter rejection."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _reject(code: str) -> ManualResearchAdapterRejected:
    return ManualResearchAdapterRejected(code)


def _public_error_code(code: str) -> str:
    if any(token in _FORBIDDEN_ERROR_TOKENS for token in code.split("_")):
        return "UNAUTHORIZED_INPUT"
    return code


def _path(value: Path | str, *, label: str) -> Path:
    if not isinstance(value, (Path, str)):
        raise _reject("ARGUMENTS_INVALID")
    raw = os.fspath(value)
    if not raw or "\x00" in raw:
        raise _reject("PATH_UNSAFE")
    candidate = Path(raw)
    if not candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise _reject("PATH_UNSAFE")
    current = candidate
    while True:
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise _reject("INPUT_MISSING") from exc
        except OSError as exc:
            raise _reject("PATH_UNSAFE") from exc
        if stat.S_ISLNK(info.st_mode):
            raise _reject("PATH_UNSAFE")
        if current != candidate and not stat.S_ISDIR(info.st_mode):
            raise _reject("PATH_UNSAFE")
        if current == Path("/"):
            break
        current = current.parent
    return candidate


def _directory(value: Path | str, *, label: str) -> Path:
    path = _path(value, label=label)
    try:
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise _reject("PATH_UNSAFE")
    except FileNotFoundError as exc:
        raise _reject("INPUT_MISSING") from exc
    return path


def _regular_file(value: Path | str, *, label: str, max_bytes: int) -> tuple[Path, bytes]:
    path = _path(value, label=label)
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError as exc:
        raise _reject("INPUT_MISSING") from exc
    except OSError as exc:
        raise _reject("PATH_UNSAFE") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise _reject("PATH_UNSAFE")
        if info.st_size > max_bytes:
            raise _reject("INPUT_TOO_LARGE")
        chunks: list[bytes] = []
        remaining = max_bytes
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining == 0 and os.read(descriptor, 1):
            raise _reject("INPUT_TOO_LARGE")
        return path, b"".join(chunks)
    except OSError as exc:
        raise _reject("INPUT_MISSING") from exc
    finally:
        os.close(descriptor)


def _hash(value: Any) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise _reject("ARGUMENTS_INVALID")
    return value


def _metadata(path: Path | str, *, label: str) -> Mapping[str, Any]:
    _, raw = _regular_file(path, label=label, max_bytes=_MAX_METADATA_BYTES)
    try:
        value = parse_canonical_json(raw, max_bytes=_MAX_METADATA_BYTES)
    except Exception as exc:
        raise _reject("ARGUMENTS_INVALID") from exc
    if not isinstance(value, Mapping):
        raise _reject("ARGUMENTS_INVALID")
    return value


def _dataclass_metadata(path: Path | str, *, label: str, type_: Any) -> Any:
    value = _metadata(path, label=label)
    expected = {field.name for field in fields(type_)}
    if set(value) != expected:
        raise _reject("ARGUMENTS_INVALID")
    try:
        return type_(**dict(value))
    except (TypeError, ValueError) as exc:
        raise _reject("ARGUMENTS_INVALID") from exc


def _error_response(code: str) -> dict[str, str]:
    return {
        "schema_version": RESPONSE_SCHEMA,
        "status": "ERROR",
        "error_code": code,
    }


def _success_response(bundle_id: str, bundle_path: Path) -> dict[str, Any]:
    return {
        "schema_version": RESPONSE_SCHEMA,
        "status": "SUCCESS",
        "bundle_id": bundle_id,
        "bundle_path": str(bundle_path),
        "verification_status": VERIFICATION_STATUS,
    }


def invoke_manual_research_prediction(
    *,
    sealed_bundle_dir: Path | str,
    run_dir: Path | str,
    evidence_expectations: Path | str,
    sealing_identity: Path | str,
    embedded_form: Path | str,
    form_sha256: str,
    model: Path | str,
    model_manifest: Path | str,
    model_sha256: str,
    model_manifest_sha256: str,
    config: Path | str,
    config_sha256: str,
    scoring_identity: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """Invoke GHU-053 from explicit caller-owned, hash-pinned artifacts."""

    bundle_dir = _directory(sealed_bundle_dir, label="sealed_bundle_dir")
    isolated_run_dir = _directory(run_dir, label="run_dir")
    isolated_output_root = _directory(output_root, label="output_root")
    expected = _dataclass_metadata(
        evidence_expectations, label="evidence_expectations", type_=SealExpectations
    )
    evidence_identity = _dataclass_metadata(
        sealing_identity, label="sealing_identity", type_=SealingIdentity
    )
    implementation_identity = _dataclass_metadata(
        scoring_identity, label="scoring_identity", type_=ResearchScoringIdentity
    )
    form_sha256 = _hash(form_sha256)
    model_sha256 = _hash(model_sha256)
    model_manifest_sha256 = _hash(model_manifest_sha256)
    config_sha256 = _hash(config_sha256)
    _, form_bytes = _regular_file(
        embedded_form, label="embedded_form", max_bytes=_MAX_FORM_BYTES
    )
    model_path, model_bytes = _regular_file(model, label="model", max_bytes=_MAX_MODEL_BYTES)
    manifest_path, manifest_bytes = _regular_file(
        model_manifest, label="model_manifest", max_bytes=_MAX_METADATA_BYTES
    )
    _config_path, config_bytes = _regular_file(
        config, label="config", max_bytes=_MAX_CONFIG_BYTES
    )
    if hashlib.sha256(form_bytes).hexdigest() != form_sha256:
        raise _reject("FORM_HASH_MISMATCH")
    if hashlib.sha256(model_bytes).hexdigest() != model_sha256:
        raise _reject("MODEL_HASH_MISMATCH")
    if hashlib.sha256(manifest_bytes).hexdigest() != model_manifest_sha256:
        raise _reject("MODEL_MANIFEST_HASH_MISMATCH")
    if hashlib.sha256(config_bytes).hexdigest() != config_sha256:
        raise _reject("CONFIG_HASH_MISMATCH")
    try:
        frozen_model = load_frozen_model(model_path, manifest_path)
    except (OSError, ValueError, ResidualContractError, KeyError, TypeError) as exc:
        raise _reject("MODEL_INVALID") from exc
    try:
        result = score_verified_manual_evidence(
            sealed_bundle_dir=bundle_dir,
            run_dir=isolated_run_dir,
            evidence_expected=expected,
            evidence_identity=evidence_identity,
            embedded_form_bytes=form_bytes,
            form_sha256=form_sha256,
            config_bytes=config_bytes,
            config_sha256=config_sha256,
            frozen_model=frozen_model,
            expected_model_sha256=model_sha256,
            expected_model_manifest_sha256=model_manifest_sha256,
            scoring_identity=implementation_identity,
            output_root=isolated_output_root,
        )
    except (ManualEvidenceRejected, ManualResearchScoringRejected) as exc:
        raise _reject(_public_error_code(exc.code)) from exc
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise _reject("INPUT_UNVERIFIED") from exc
    except OSError as exc:
        raise _reject("OUTPUT_ROOT_UNSAFE") from exc
    return _success_response(result.prediction["bundle_id"], result.bundle_dir)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="manual-research-predict",
        description="Invoke the isolated GHU-053 manual research scorer.",
        allow_abbrev=False,
    )
    for name in (
        "sealed-bundle-dir",
        "run-dir",
        "evidence-expectations",
        "sealing-identity",
        "embedded-form",
        "model",
        "model-manifest",
        "config",
        "scoring-identity",
        "output-root",
    ):
        parser.add_argument(f"--{name}", dest=name.replace("-", "_"), required=True, type=Path)
    for name in ("form-sha256", "model-sha256", "model-manifest-sha256", "config-sha256"):
        parser.add_argument(f"--{name}", dest=name.replace("-", "_"), required=True)
    return parser


def _response_bytes(response: Mapping[str, Any]) -> bytes:
    return canonical_bytes(response)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        response = invoke_manual_research_prediction(**vars(args))
    except ManualResearchAdapterRejected as exc:
        response = _error_response(exc.code)
    except (ValueError, TypeError):
        response = _error_response("ARGUMENTS_INVALID")
    sys.stdout.buffer.write(_response_bytes(response))
    return 0 if response["status"] == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RESPONSE_SCHEMA",
    "VERIFICATION_STATUS",
    "ManualResearchAdapterRejected",
    "invoke_manual_research_prediction",
    "main",
]
