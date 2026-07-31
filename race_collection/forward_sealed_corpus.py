"""Append-only prospective capture for authentic Phase 7 training evidence."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from scripts.ingest_results_for_date import (
    finish_positions_follow_competition_ranking,
    parse_thedogs_result_html_runner_rows,
)

from .artifacts import ArtifactStoreError, LocalArtifactStore
from .domain import ArtifactChecksum, require_aware
from .features import FeatureQuarantine, derive_features
from .model_bundle import SUPPORTED_FEATURE_CONTRACT

FORWARD_CORPUS_ORIGIN = "official-result-first-observation-v1"
SOURCE_CAPTURE_SCHEMA = "forward-source-capture-v1"
RESULT_SCHEMA = "official-result-normalized-v1"
PREJUMP_RECEIPT_SCHEMA = "forward-prejump-receipt-v1"
RESULT_RECEIPT_SCHEMA = "forward-result-receipt-v1"
CLOSURE_RECEIPT_SCHEMA = "forward-race-closure-v1"
STATUS_SCHEMA = "forward-sealed-corpus-status-v1"
OFFICIAL_RESULT_SOURCE = "thedogs-official"
PREJUMP_SOURCE = "thedogs-race-card"
RESPONSE_STAGE_SCHEMA = "official-result-response-stage-v1"

_RESULT_DERIVED_KEYS = {
    "finish_order",
    "finish_position",
    "place",
    "position",
    "result",
    "result_order",
    "winner",
}


class ForwardCorpusRejected(ValueError):
    """Prospective evidence cannot be truthfully captured or closed."""


@dataclass(frozen=True, slots=True)
class ClosedPackage:
    package_bytes: bytes
    package_checksum: ArtifactChecksum
    manifest_checksum: ArtifactChecksum
    artifacts: Mapping[str, bytes]


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ForwardCorpusRejected("value is not exact JSON") from error


def _checksum(content: bytes) -> ArtifactChecksum:
    return LocalArtifactStore.checksum(content)


def _canonical_object(content: bytes, name: str) -> dict[str, Any]:
    if type(content) is not bytes:
        raise ForwardCorpusRejected(f"{name} must be exact bytes")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ForwardCorpusRejected(f"{name} is not valid JSON") from error
    if type(value) is not dict or canonical_json(value) != content:
        raise ForwardCorpusRejected(f"{name} must be one canonical JSON object")
    return value


def _known_text(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value.casefold() == "unknown"
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise ForwardCorpusRejected(f"{name} is missing or ambiguous")
    return value


def _identity_key(value: Any, name: str) -> str:
    identity = _known_text(value, name)
    normalized = unicodedata.normalize("NFKC", " ".join(identity.split())).casefold()
    if not normalized:
        raise ForwardCorpusRejected(f"{name} is missing or ambiguous")
    return normalized


def _timestamp(value: Any, name: str) -> tuple[datetime, str]:
    if type(value) is not str:
        raise ForwardCorpusRejected(f"{name} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value)
        require_aware(parsed, name)
    except (TypeError, ValueError) as error:
        raise ForwardCorpusRejected(f"{name} timestamp is invalid") from error
    return parsed, parsed.isoformat(timespec="microseconds")


def _source_url(
    value: Any,
    name: str,
    *,
    allow_official_result_query: bool = False,
) -> str:
    url = _known_text(value, name)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.hostname not in {"www.thedogs.com.au", "thedogs.com.au"}
        or parsed.username is not None
        or parsed.password is not None
        or (
            parsed.query
            and not (
                allow_official_result_query
                and parsed.query == "trial=false"
                and parsed.path.endswith("/results")
            )
        )
        or parsed.fragment
    ):
        raise ForwardCorpusRejected(f"{name} is not a canonical source URL")
    return url


def _reject_result_derived(value: Any) -> None:
    if type(value) is dict:
        for key, nested in value.items():
            if type(key) is str and key.casefold() in _RESULT_DERIVED_KEYS:
                raise ForwardCorpusRejected("pre-jump evidence contains a result-derived field")
            _reject_result_derived(nested)
    elif type(value) is list:
        for nested in value:
            _reject_result_derived(nested)


def _metadata(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict or not value:
        raise ForwardCorpusRejected(f"{name} must be a non-empty object")
    canonical_json(value)
    _reject_result_derived(value)
    return dict(value)


def _runner_rows(values: Any) -> tuple[dict[str, Any], ...]:
    if type(values) not in {list, tuple} or len(values) < 2:
        raise ForwardCorpusRejected("source-native runner identities are incomplete")
    rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    for value in values:
        if type(value) is not dict or set(value) != {
            "source_native_runner_id",
            "name",
            "box_number",
        }:
            raise ForwardCorpusRejected("source-native runner identity envelope is invalid")
        runner_id = _known_text(value["source_native_runner_id"], "source-native runner id")
        name = _known_text(value["name"], "runner name")
        key = _identity_key(runner_id, "source-native runner id")
        if key in keys:
            raise ForwardCorpusRejected("duplicate normalized source-native runner identity")
        keys.add(key)
        box_number = value["box_number"]
        if type(box_number) is not int or not 1 <= box_number <= 20:
            raise ForwardCorpusRejected("runner box/rug identity is invalid")
        if any(row["box_number"] == box_number for row in rows):
            raise ForwardCorpusRejected("duplicate runner box/rug identity")
        rows.append(
            {
                "source_native_runner_id": runner_id,
                "name": name,
                "box_number": box_number,
            }
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: _identity_key(
                row["source_native_runner_id"], "source-native runner id"
            ),
        )
    )


def _bounded_id(value: Any, name: str) -> str:
    result = _known_text(value, name)
    if len(result.encode()) > 128:
        raise ForwardCorpusRejected(f"{name} is too long")
    return result


def _normalization_identity() -> tuple[str, str, str]:
    parser_hash = str(_checksum(inspect.getsource(parse_thedogs_result_html_runner_rows).encode()))
    schema_hash = str(_checksum(RESULT_SCHEMA.encode()))
    implementation = (
        inspect.getsource(_normalize_official_result).encode()
        + inspect.getsource(parse_thedogs_result_html_runner_rows).encode()
        + inspect.getsource(finish_positions_follow_competition_ranking).encode()
        + RESULT_SCHEMA.encode()
    )
    return parser_hash, schema_hash, str(_checksum(implementation))


def _normalize_official_result(
    raw_response_bytes: bytes,
    *,
    race_id: str,
    frozen_runners: Sequence[Mapping[str, Any]],
) -> bytes:
    if type(raw_response_bytes) is not bytes or not raw_response_bytes:
        raise ForwardCorpusRejected("immutable official-result response bytes are required")
    try:
        markup = raw_response_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ForwardCorpusRejected("official result is not supported UTF-8 HTML") from error
    try:
        parsed = parse_thedogs_result_html_runner_rows(markup)
    except (TypeError, ValueError) as error:
        raise ForwardCorpusRejected("official result parser rejected response") from error
    if not parsed:
        raise ForwardCorpusRejected("official result HTML contains no result rows")
    by_box = {row["box_number"]: row for row in parsed if type(row) is dict}
    if len(by_box) != len(parsed) or set(by_box) != {
        runner["box_number"] for runner in frozen_runners
    }:
        raise ForwardCorpusRejected("official result runner box/rug identity mismatch")
    normalized = []
    positions = []
    for runner in sorted(frozen_runners, key=lambda item: item["box_number"]):
        parsed_row = by_box[runner["box_number"]]
        if parsed_row.get("dog_name") != runner["name"]:
            raise ForwardCorpusRejected("official result runner name identity mismatch")
        position = parsed_row.get("finish_position")
        status = parsed_row.get("status")
        if (position is None) == (status is None):
            raise ForwardCorpusRejected("official finish/status combination is inconsistent")
        if position is not None:
            if type(position) is not int or not 1 <= position <= len(frozen_runners):
                raise ForwardCorpusRejected("official finish position is out of range")
            positions.append(position)
        elif status not in {"SCRATCHED", "DNF", "DQ"}:
            raise ForwardCorpusRejected("official terminal status is unsupported")
        normalized.append(
            {
                "source_native_runner_id": runner["source_native_runner_id"],
                "box_number": runner["box_number"],
                "name": runner["name"],
                "finish_position": position,
                "status": status,
            }
        )
    if not finish_positions_follow_competition_ranking(positions):
        raise ForwardCorpusRejected("official finishes do not use competition ranking")
    runner_set_hash = str(_checksum(canonical_json(list(frozen_runners))))
    return canonical_json(
        {
            "schema_version": RESULT_SCHEMA,
            "race_id": race_id,
            "runner_set_hash": runner_set_hash,
            "runners": normalized,
        }
    )


class ForwardSealedCorpus:
    """Immutable object store plus append-only per-race stage receipts."""

    def __init__(
        self,
        root: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ):
        self.root = Path(root).resolve()
        self.artifacts = LocalArtifactStore(self.root / "artifacts")
        self._clock = clock or (lambda: datetime.now().astimezone())

    def _race_directory(self, race_id: str) -> Path:
        identity = _identity_key(race_id, "race_id").encode()
        return self.root / "races" / hashlib.sha256(identity).hexdigest()

    def _receipt_path(self, race_id: str, stage: str) -> Path:
        return self._race_directory(race_id) / f"{stage}.json"

    @staticmethod
    def _publish_once(path: Path, content: bytes) -> bool:
        """Publish once without an overwrite window; exact retries are no-ops."""
        try:
            anchor = next(parent for parent in path.parents if parent.name in {"races", "packages"})
        except StopIteration as error:
            raise ForwardCorpusRejected("receipt path escapes the corpus root") from error
        root = anchor.parent
        root.mkdir(parents=True, exist_ok=True)
        current = root
        for part in path.parent.relative_to(root).parts:
            current /= part
            if current.is_symlink():
                raise ForwardCorpusRejected("receipt directories must not be symlinks")
            current.mkdir(exist_ok=True)
        if path.exists():
            if path.is_symlink() or path.read_bytes() != content:
                raise ForwardCorpusRejected(f"append-only receipt conflict: {path.name}")
            return False
        descriptor, temporary_name = tempfile.mkstemp(prefix=".incoming-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.is_symlink() or path.read_bytes() != content:
                    raise ForwardCorpusRejected(f"append-only receipt conflict: {path.name}")
                return False
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return True
        finally:
            temporary.unlink(missing_ok=True)

    def _load_receipt(self, race_id: str, stage: str) -> dict[str, Any] | None:
        path = self._receipt_path(race_id, stage)
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return None
        if path.is_symlink():
            raise ForwardCorpusRejected(f"{stage} receipt must not be a symlink")
        return _canonical_object(content, f"{stage} receipt")

    def _read_artifact(self, value: Any, name: str) -> bytes:
        if type(value) is not str:
            raise ForwardCorpusRejected(f"{name} checksum is invalid")
        try:
            return self.artifacts.read(ArtifactChecksum(value))
        except (ArtifactStoreError, ValueError) as error:
            raise ForwardCorpusRejected(f"{name} bytes are missing or have hash drift") from error

    def capture_prejump(
        self,
        *,
        race_id: str,
        racing_date: str,
        raw_source_bytes: bytes,
        sealed_evidence_bytes: bytes,
        feature_schema_bytes: bytes,
        missingness_policy_bytes: bytes,
        source_name: str,
        canonical_source_url: str,
        source_native_race_id: str,
        runners: Sequence[Mapping[str, str]],
        meeting_metadata: Mapping[str, Any],
        race_metadata: Mapping[str, Any],
        source_observed_at: str,
        feature_frozen_at: str,
        scheduled_jump_at: str,
    ) -> dict[str, Any]:
        """Capture source and production feature evidence without an outcome input."""
        race_id = _known_text(race_id, "race_id")
        if source_name != PREJUMP_SOURCE:
            raise ForwardCorpusRejected("source name is not the canonical TheDogs race-card source")
        source_name = _known_text(source_name, "source name")
        canonical_source_url = _source_url(canonical_source_url, "canonical source URL")
        source_native_race_id = _known_text(source_native_race_id, "source-native race identity")
        runner_rows = _runner_rows(runners)
        meeting = _metadata(meeting_metadata, "meeting metadata")
        race = _metadata(race_metadata, "race metadata")
        if type(raw_source_bytes) is not bytes or not raw_source_bytes:
            raise ForwardCorpusRejected("immutable raw source bytes are required")

        observed_at, observed_text = _timestamp(source_observed_at, "source observed_at")
        frozen_at, frozen_text = _timestamp(feature_frozen_at, "feature frozen_at")
        jump_at, jump_text = _timestamp(scheduled_jump_at, "scheduled jump")
        if not observed_at <= frozen_at < jump_at:
            raise ForwardCorpusRejected("pre-jump source/feature temporal order is invalid")
        try:
            race_day = date.fromisoformat(racing_date)
        except (TypeError, ValueError) as error:
            raise ForwardCorpusRejected("racing date is invalid") from error
        if race_day != jump_at.date():
            raise ForwardCorpusRejected("racing date and scheduled jump disagree")

        evidence = _canonical_object(sealed_evidence_bytes, "sealed race evidence")
        schema = _canonical_object(feature_schema_bytes, "feature schema")
        policy = _canonical_object(missingness_policy_bytes, "missingness policy")
        _reject_result_derived(evidence)
        bundle_id = _known_text(schema.get("bundle_id"), "target bundle identity")
        if (
            set(evidence)
            != {
                "schema_version",
                "normalization_version",
                "race_id",
                "fields",
                "field_provenance",
                "freeze",
            }
            or set(schema)
            != {
                "bundle_id",
                "contract_version",
                "evidence_schema_version",
                "normalization_version",
                "fields",
            }
            or set(policy) != {"bundle_id", "feature_contract_version", "imputation"}
            or schema.get("contract_version") != SUPPORTED_FEATURE_CONTRACT
            or policy.get("feature_contract_version") != SUPPORTED_FEATURE_CONTRACT
            or policy.get("bundle_id") != bundle_id
            or evidence.get("race_id") != race_id
            or evidence.get("schema_version") != schema.get("evidence_schema_version")
            or evidence.get("normalization_version") != schema.get("normalization_version")
        ):
            raise ForwardCorpusRejected("sealed evidence identity or feature schema disagrees")
        contract_fields = schema.get("fields")
        if type(contract_fields) is not list or not contract_fields:
            raise ForwardCorpusRejected("feature schema fields are invalid")
        for field in contract_fields:
            if type(field) is not dict or set(field) != (
                {"name", "source_field", "semantics", "encoded_value"}
                if field.get("semantics") == "inapplicable"
                else {"name", "source_field", "semantics"}
            ):
                raise ForwardCorpusRejected("feature schema field envelope is unsupported")
        freeze = evidence.get("freeze")
        try:
            if set(freeze) != {"at", "authority", "odds_checksum"}:
                raise ForwardCorpusRejected("sealed evidence freeze provenance is incomplete")
            evidence_freeze, _ = _timestamp(freeze["at"], "sealed evidence freeze")
            _known_text(freeze["authority"], "sealed evidence freeze authority")
            ArtifactChecksum(freeze["odds_checksum"])
        except (KeyError, TypeError, ValueError) as error:
            raise ForwardCorpusRejected(
                "sealed evidence freeze provenance is incomplete"
            ) from error
        if evidence_freeze != frozen_at:
            raise ForwardCorpusRejected("sealed evidence and feature freeze timestamps disagree")
        fields = evidence.get("fields")
        if type(fields) is not dict:
            raise ForwardCorpusRejected("sealed evidence fields are incomplete")
        runner_ids = [row["source_native_runner_id"] for row in runner_rows]
        if fields.get("runner_set") != runner_ids:
            raise ForwardCorpusRejected(
                "sealed evidence and source-native runner identities disagree"
            )
        identities = fields.get("runner_identity")
        if (
            type(identities) is not dict
            or set(identities) != set(runner_ids)
            or any(identities[runner_id] != "authoritative" for runner_id in runner_ids)
        ):
            raise ForwardCorpusRejected("sealed evidence runner identity is ambiguous")

        raw_source = self.artifacts.put(
            raw_source_bytes, media_type="application/octet-stream"
        ).checksum
        provenance = evidence.get("field_provenance")
        required_bindings = {"runner_set", "runner_identity", "runner_features"}
        if type(provenance) is not list or not provenance:
            raise ForwardCorpusRejected("sealed feature source bindings are incomplete")
        bound_fields = set()
        for item in provenance:
            if type(item) is not dict or set(item) != {
                "field",
                "authority",
                "critical",
                "value",
                "source",
                "artifact_checksum",
            }:
                raise ForwardCorpusRejected("sealed feature source binding is invalid")
            try:
                ArtifactChecksum(item["artifact_checksum"])
            except (KeyError, ValueError) as error:
                raise ForwardCorpusRejected(
                    "sealed feature source binding checksum is invalid"
                ) from error
            _known_text(item.get("field"), "sealed feature source binding field")
            _known_text(item.get("authority"), "sealed feature source binding authority")
            _known_text(item.get("source"), "sealed feature source binding source")
            if type(item.get("critical")) is not bool:
                raise ForwardCorpusRejected("sealed feature source binding criticality is invalid")
            if (
                item.get("field") in required_bindings
                and item.get("artifact_checksum") == str(raw_source)
                and item.get("source") == source_name
                and item.get("value") == fields.get(item["field"])
            ):
                bound_fields.add(item["field"])
        if bound_fields != required_bindings:
            raise ForwardCorpusRejected(
                "runner feature evidence is not bound to the preserved raw source bytes"
            )
        sealed_evidence = self.artifacts.put(
            sealed_evidence_bytes, media_type="application/json"
        ).checksum
        feature_schema = self.artifacts.put(
            feature_schema_bytes, media_type="application/json"
        ).checksum
        missingness_policy = self.artifacts.put(
            missingness_policy_bytes, media_type="application/json"
        ).checksum
        try:
            derived = derive_features(
                sealed_evidence_bytes,
                expected_evidence_checksum=sealed_evidence,
                schema_bytes=feature_schema_bytes,
                expected_schema_checksum=feature_schema,
                missingness_policy_bytes=missingness_policy_bytes,
                expected_missingness_checksum=missingness_policy,
            )
        except (FeatureQuarantine, ValueError) as error:
            raise ForwardCorpusRejected(str(error)) from error
        if list(derived.matrix.runner_ids) != runner_ids:
            raise ForwardCorpusRejected(
                "feature matrix and source-native runner identities disagree"
            )
        matrix_bytes = canonical_json(
            {
                "runner_ids": list(derived.matrix.runner_ids),
                "columns": list(derived.matrix.columns),
                "rows": [list(row) for row in derived.matrix.rows],
            }
        )
        feature_matrix = self.artifacts.put(
            matrix_bytes,
            media_type="application/json",
            expected_checksum=derived.matrix.checksum,
        ).checksum

        source_capture_bytes = canonical_json(
            {
                "schema_version": SOURCE_CAPTURE_SCHEMA,
                "race_id": race_id,
                "racing_date": racing_date,
                "source_name": source_name,
                "canonical_source_url": canonical_source_url,
                "source_native_race_id": source_native_race_id,
                "meeting_metadata": meeting,
                "race_metadata": race,
                "scheduled_jump_at": jump_text,
                "source_observed_at": observed_text,
                "feature_frozen_at": frozen_text,
                "raw_source_checksum": str(raw_source),
                "sealed_evidence_checksum": str(sealed_evidence),
                "runners": list(runner_rows),
                "identity_authority": "source-native",
                "reconstructed": False,
            }
        )
        source_capture = self.artifacts.put(
            source_capture_bytes, media_type="application/json"
        ).checksum
        receipt = {
            "schema_version": PREJUMP_RECEIPT_SCHEMA,
            "race_id": race_id,
            "racing_date": racing_date,
            "target_bundle_id": bundle_id,
            "source_native_race_id": source_native_race_id,
            "runner_ids": runner_ids,
            "source_observed_at": observed_text,
            "feature_frozen_at": frozen_text,
            "scheduled_jump_at": jump_text,
            "raw_source_checksum": str(raw_source),
            "source_checksum": str(sealed_evidence),
            "source_capture_checksum": str(source_capture),
            "feature_schema_checksum": str(feature_schema),
            "missingness_policy_checksum": str(missingness_policy),
            "feature_matrix_checksum": str(feature_matrix),
        }
        existing_receipt = self._load_receipt(race_id, "prejump")
        if existing_receipt is None:
            captured_at = self._clock()
            try:
                if type(captured_at) is not datetime:
                    raise TypeError
                require_aware(captured_at, "collector pre-jump capture")
            except (TypeError, ValueError) as error:
                raise ForwardCorpusRejected(
                    "collector pre-jump capture timestamp is invalid"
                ) from error
            if not frozen_at <= captured_at < jump_at:
                raise ForwardCorpusRejected(
                    "collector did not publish the pre-jump receipt prospectively"
                )
        self._publish_once(self._receipt_path(race_id, "prejump"), canonical_json(receipt))
        self.artifacts.put(canonical_json(receipt), media_type="application/json")
        return receipt

    def _result_observations(self, race_id: str) -> list[dict[str, Any]]:
        directory = self._race_directory(race_id) / "result-observations"
        if not directory.exists():
            return []
        observations = []
        for path in sorted(directory.glob("*.json")):
            if path.is_symlink():
                raise ForwardCorpusRejected("result observation receipt must not be a symlink")
            observation = _canonical_object(path.read_bytes(), "result observation receipt")
            self._read_artifact(
                observation.get("raw_result_checksum"),
                "observed raw official result",
            )
            observations.append(observation)
        return observations

    def _publication_timestamp_capture_result(
        self,
        *,
        race_id: str,
        raw_result_bytes: bytes,
        source_name: str,
        canonical_source_url: str,
        source_native_race_id: str,
        runners: Sequence[Mapping[str, str]],
        official_order: Sequence[str],
        result_observed_at: str,
        result_published_at: str | None,
        publication_timestamp_status: str,
    ) -> dict[str, Any]:
        """Capture an official result and close only with a source-declared timestamp."""
        raise ForwardCorpusRejected(
            "legacy publication-timestamp capture cannot create new-origin artifacts"
        )
        # Retained below solely so existing legacy bytes remain structurally readable.
        race_id = _known_text(race_id, "race_id")
        pre = self._load_receipt(race_id, "prejump")
        if pre is None:
            raise ForwardCorpusRejected("pre-jump evidence must exist before result capture")
        existing_result = self._load_receipt(race_id, "result")
        if existing_result is not None and publication_timestamp_status != "source-declared":
            raise ForwardCorpusRejected("closed result cannot accept another observation")
        if existing_result is not None:
            for field in ("raw_result_checksum", "official_result_checksum"):
                self._read_artifact(existing_result.get(field), field)
        for field in (
            "raw_source_checksum",
            "source_checksum",
            "source_capture_checksum",
            "feature_schema_checksum",
            "missingness_policy_checksum",
            "feature_matrix_checksum",
        ):
            self._read_artifact(pre[field], field)
        if type(raw_result_bytes) is not bytes or not raw_result_bytes:
            raise ForwardCorpusRejected("immutable official-result source bytes are required")
        source_name = _known_text(source_name, "official result source")
        canonical_source_url = _source_url(
            canonical_source_url, "official result canonical source URL"
        )
        source_native_race_id = _known_text(
            source_native_race_id, "official result source-native race identity"
        )
        if source_native_race_id != pre["source_native_race_id"]:
            raise ForwardCorpusRejected("source-native race identity mismatch")
        runner_rows = _runner_rows(runners)
        runner_ids = [row["source_native_runner_id"] for row in runner_rows]
        if runner_ids != pre["runner_ids"]:
            raise ForwardCorpusRejected("official result runner drift")
        source_capture = _canonical_object(
            self._read_artifact(pre["source_capture_checksum"], "source capture"),
            "source capture",
        )
        if list(runner_rows) != source_capture.get("runners"):
            raise ForwardCorpusRejected("official result runner identity or name drift")
        if type(official_order) not in {list, tuple}:
            raise ForwardCorpusRejected("official result order is incomplete")
        ordered = [_known_text(value, "official runner identity") for value in official_order]
        if len(ordered) != len(set(ordered)) or set(ordered) != set(runner_ids):
            raise ForwardCorpusRejected("official result and pre-jump runner sets disagree")

        jump_at, _ = _timestamp(pre["scheduled_jump_at"], "scheduled jump")
        observed_at, observed_text = _timestamp(result_observed_at, "result observed_at")
        if not jump_at < observed_at:
            raise ForwardCorpusRejected("official result observation must be after jump")
        raw_checksum = _checksum(raw_result_bytes)
        if existing_result is not None and existing_result.get("raw_result_checksum") != str(
            raw_checksum
        ):
            raise ForwardCorpusRejected("official result raw-byte hash drift")
        prior_observations = self._result_observations(race_id)
        if prior_observations and any(
            item["raw_result_checksum"] != str(raw_checksum) for item in prior_observations
        ):
            raise ForwardCorpusRejected("official result raw-byte hash drift")
        if prior_observations and any(
            item[field] != expected
            for item in prior_observations
            for field, expected in (
                ("source_name", source_name),
                ("canonical_source_url", canonical_source_url),
                ("source_native_race_id", source_native_race_id),
                ("runner_ids", runner_ids),
                ("official_order", ordered),
            )
        ):
            raise ForwardCorpusRejected("official result observation provenance drift")

        if publication_timestamp_status == "not-exposed-by-source":
            if result_published_at is not None:
                raise ForwardCorpusRejected(
                    "unavailable result publication timestamp cannot have a value"
                )
            raw_result = self.artifacts.put(
                raw_result_bytes, media_type="application/octet-stream"
            ).checksum
            observation = {
                "schema_version": "forward-result-observation-v1",
                "race_id": race_id,
                "source_name": source_name,
                "canonical_source_url": canonical_source_url,
                "source_native_race_id": source_native_race_id,
                "runner_ids": runner_ids,
                "official_order": ordered,
                "result_observed_at": observed_text,
                "result_published_at": None,
                "publication_timestamp_status": publication_timestamp_status,
                "raw_result_checksum": str(raw_result),
                "identity_authority": "source-native",
                "reconstructed": False,
                "closure_decision": "BLOCKED_RESULT_PUBLICATION_TIMESTAMP",
            }
            content = canonical_json(observation)
            path = (
                self._race_directory(race_id)
                / "result-observations"
                / f"{_checksum(content).hex_digest}.json"
            )
            self._publish_once(path, content)
            return observation
        if publication_timestamp_status != "source-declared":
            raise ForwardCorpusRejected("result publication timestamp status is unsupported")
        published_at, published_text = _timestamp(result_published_at, "result published_at")
        if not jump_at < published_at <= observed_at:
            raise ForwardCorpusRejected("official result temporal order is invalid")

        raw_result = self.artifacts.put(
            raw_result_bytes, media_type="application/octet-stream"
        ).checksum
        result_bytes = canonical_json(
            {
                "schema_version": RESULT_SCHEMA,
                "race_id": race_id,
                "official": True,
                "order": ordered,
                "published_at": published_text,
                "exclusions": [],
                "runner_names": {
                    row["source_native_runner_id"]: row["name"] for row in runner_rows
                },
                "provenance": {
                    "source": source_name,
                    "canonical_source_url": canonical_source_url,
                    "source_native_race_id": source_native_race_id,
                    "observed_at": observed_text,
                    "publication_timestamp_status": "source-declared",
                    "raw_result_checksum": str(raw_result),
                    "identity_authority": "source-native",
                    "reconstructed": False,
                },
            }
        )
        official_result = self.artifacts.put(result_bytes, media_type="application/json").checksum
        result_receipt = {
            "schema_version": RESULT_RECEIPT_SCHEMA,
            "race_id": race_id,
            "runner_ids": runner_ids,
            "official_order": ordered,
            "result_published_at": published_text,
            "result_observed_at": observed_text,
            "raw_result_checksum": str(raw_result),
            "official_result_checksum": str(official_result),
        }
        self._publish_once(self._receipt_path(race_id, "result"), canonical_json(result_receipt))
        return self._close(pre, result_receipt)

    def _close(
        self,
        pre: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        race_id = pre["race_id"]
        if result["race_id"] != race_id or result["runner_ids"] != pre["runner_ids"]:
            raise ForwardCorpusRejected("race closure identity mismatch")
        for field in (
            "raw_result_checksum",
            "official_result_checksum",
        ):
            self._read_artifact(result[field], field)
        training_example_id = (
            "forward-sealed-"
            + hashlib.sha256(f"{race_id}\0{pre['source_native_race_id']}".encode()).hexdigest()
        )
        race_entry = {
            "training_example_id": training_example_id,
            "race_id": race_id,
            "racing_date": pre["racing_date"],
            "source_checksum": pre["source_checksum"],
            "source_capture_checksum": pre["source_capture_checksum"],
            "raw_source_checksum": pre["raw_source_checksum"],
            "official_result_checksum": result["official_result_checksum"],
            "raw_result_checksum": result["raw_result_checksum"],
            "feature_matrix_checksum": pre["feature_matrix_checksum"],
            "runner_ids": pre["runner_ids"],
            "source_observed_at": pre["source_observed_at"],
            "feature_observed_at": pre["feature_frozen_at"],
            "scheduled_jump_at": pre["scheduled_jump_at"],
            "result_published_at": result["result_published_at"],
            "result_observed_at": result["result_observed_at"],
        }
        artifact_bytes = canonical_json(
            {
                "schema_version": "historical-training-example-v1",
                "origin": FORWARD_CORPUS_ORIGIN,
                "forward_sealed": True,
                "promotion_evidence_eligible": False,
                **race_entry,
                "official_order": result["official_order"],
            }
        )
        artifact_checksum = self.artifacts.put(
            artifact_bytes, media_type="application/json"
        ).checksum
        race_entry["artifact_checksum"] = str(artifact_checksum)
        closure = {
            "schema_version": CLOSURE_RECEIPT_SCHEMA,
            "race_id": race_id,
            "target_bundle_id": pre["target_bundle_id"],
            "feature_schema_checksum": pre["feature_schema_checksum"],
            "missingness_policy_checksum": pre["missingness_policy_checksum"],
            "race": race_entry,
        }
        self._publish_once(self._receipt_path(race_id, "closure"), canonical_json(closure))
        return closure

    def _closures(self) -> list[dict[str, Any]]:
        directory = self.root / "races"
        if not directory.exists():
            return []
        closures = []
        for path in sorted(directory.glob("*/closure.json")):
            if path.is_symlink():
                raise ForwardCorpusRejected("closure receipt must not be a symlink")
            closures.append(_canonical_object(path.read_bytes(), "closure receipt"))
        return sorted(
            closures,
            key=lambda item: _identity_key(item["race_id"], "race_id"),
        )

    def _publication_timestamp_build_package(self) -> ClosedPackage:
        """Build and persist one deterministic package from every closed race."""
        closures = self._closures()
        if not closures:
            raise ForwardCorpusRejected("no closed races are available")
        bundle_ids = {item["target_bundle_id"] for item in closures}
        schema_checksums = {item["feature_schema_checksum"] for item in closures}
        policy_checksums = {item["missingness_policy_checksum"] for item in closures}
        if len(bundle_ids) != 1 or len(schema_checksums) != 1 or len(policy_checksums) != 1:
            raise ForwardCorpusRejected("closed races use different feature contracts")
        races = [item["race"] for item in closures]
        manifest = {
            "schema_version": "historical-source-manifest-v1",
            "corpus_origin": FORWARD_CORPUS_ORIGIN,
            "target_bundle_id": next(iter(bundle_ids)),
            "feature_schema_checksum": next(iter(schema_checksums)),
            "missingness_policy_checksum": next(iter(policy_checksums)),
            "races": races,
        }
        manifest_bytes = canonical_json(manifest)
        manifest_checksum = _checksum(manifest_bytes)
        package_bytes = canonical_json(
            {
                "schema_version": "historical-source-package-v1",
                "manifest_checksum": str(manifest_checksum),
                "manifest": manifest,
            }
        )
        package_checksum = _checksum(package_bytes)
        declared = {
            str(manifest["feature_schema_checksum"]),
            str(manifest["missingness_policy_checksum"]),
            *{
                race[field]
                for race in races
                for field in (
                    "source_checksum",
                    "source_capture_checksum",
                    "raw_source_checksum",
                    "official_result_checksum",
                    "raw_result_checksum",
                    "feature_matrix_checksum",
                    "artifact_checksum",
                )
            },
        }
        artifacts = {
            checksum: self._read_artifact(checksum, "package artifact")
            for checksum in sorted(declared)
        }
        from .source_admission import admit_historical_source

        admit_historical_source(package_bytes, artifacts=artifacts)
        self.artifacts.put(
            package_bytes,
            media_type="application/json",
            expected_checksum=package_checksum,
        )
        package_path = self.root / "packages" / f"{manifest_checksum.hex_digest}.json"
        self._publish_once(package_path, package_bytes)
        return ClosedPackage(
            package_bytes,
            package_checksum,
            manifest_checksum,
            artifacts,
        )

    def _publication_timestamp_status(self) -> dict[str, Any]:
        """Return deterministic state while verifying all referenced immutable bytes."""
        directory = self.root / "races"
        rows = []
        if directory.exists():
            for race_directory in sorted(path for path in directory.iterdir() if path.is_dir()):
                pre_path = race_directory / "prejump.json"
                if not pre_path.exists():
                    continue
                if pre_path.is_symlink():
                    raise ForwardCorpusRejected("prejump receipt must not be a symlink")
                pre = _canonical_object(pre_path.read_bytes(), "prejump receipt")
                state = "PREJUMP_CAPTURED"
                for field in (
                    "raw_source_checksum",
                    "source_checksum",
                    "source_capture_checksum",
                    "feature_schema_checksum",
                    "missingness_policy_checksum",
                    "feature_matrix_checksum",
                ):
                    self._read_artifact(pre[field], field)
                result = self._load_receipt(pre["race_id"], "result")
                closure = self._load_receipt(pre["race_id"], "closure")
                blocked = self._result_observations(pre["race_id"])
                if result is not None:
                    for field in ("raw_result_checksum", "official_result_checksum"):
                        self._read_artifact(result[field], field)
                    state = "RESULT_CAPTURED"
                if blocked and result is None:
                    state = "BLOCKED_RESULT_PUBLICATION_TIMESTAMP"
                if closure is not None:
                    self._read_artifact(closure["race"]["artifact_checksum"], "training example")
                    state = "CLOSED"
                rows.append(
                    {
                        "race_id": pre["race_id"],
                        "state": state,
                        "result_observation_count": len(blocked),
                    }
                )
        rows.sort(key=lambda row: _identity_key(row["race_id"], "race_id"))
        return {
            "schema_version": STATUS_SCHEMA,
            "race_count": len(rows),
            "closed_race_count": sum(row["state"] == "CLOSED" for row in rows),
            "races": rows,
        }

    def _official_receipts(self, race_id: str) -> list[dict[str, Any]]:
        pre = self._load_receipt(race_id, "prejump")
        if pre is None:
            raise ForwardCorpusRejected("official history references missing pre-jump receipt")
        return [
            observation
            for _stage, observation in self._official_response_history(pre)
            if observation is not None
        ]

    def _official_response_history(
        self, pre: Mapping[str, Any]
    ) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
        race_id = pre["race_id"]
        directory = self._race_directory(race_id) / "official-requests"
        if not directory.exists():
            return []
        if directory.is_symlink() or not directory.is_dir():
            raise ForwardCorpusRejected("official request root is invalid")
        history = []
        for request_directory in sorted(directory.iterdir(), key=lambda path: path.name):
            if request_directory.is_symlink() or not request_directory.is_dir():
                raise ForwardCorpusRejected("official request directory is invalid")
            entries = list(request_directory.iterdir())
            if any(entry.is_symlink() or not entry.is_file() for entry in entries):
                raise ForwardCorpusRejected("official request artifact is invalid")
            entry_names = {entry.name for entry in entries}
            if (
                "response-stage.json" not in entry_names
                or not entry_names <= {"response-stage.json", "observation.json"}
            ):
                raise ForwardCorpusRejected("official request artifact inventory is invalid")
            stage = self._load_request_receipt(
                request_directory / "response-stage.json", "response-stage"
            )
            observation = self._load_request_receipt(
                request_directory / "observation.json", "observation"
            )
            if stage is None:
                raise ForwardCorpusRejected("official response-stage receipt is missing")
            self._verify_response_stage(pre, stage)
            if request_directory.name != hashlib.sha256(
                _bounded_id(stage.get("request_id"), "request_id").encode()
            ).hexdigest():
                raise ForwardCorpusRejected("response-stage request directory binding disagrees")
            if observation is not None:
                self._verify_observation(pre, observation)
                self._verify_observation_stage_binding(pre, observation, stage)
            history.append((stage, observation))
        request_ids = [stage["request_id"] for stage, _observation in history]
        if len(request_ids) != len(set(request_ids)):
            raise ForwardCorpusRejected("duplicate official response-stage request identity")
        return history

    def _observation_inventory(
        self,
        pre: Mapping[str, Any],
        observations: Sequence[Mapping[str, Any]],
    ) -> tuple[list[str], list[str], list[str]]:
        history = self._official_response_history(pre)
        completed = [
            observation for _stage, observation in history if observation is not None
        ]
        if list(observations) != completed:
            raise ForwardCorpusRejected(
                "official completed observation inventory is incomplete or reordered"
            )
        response_stages = [
            str(_checksum(canonical_json(stage))) for stage, _observation in history
        ]
        raw_responses = [stage["raw_response_checksum"] for stage, _observation in history]
        observation_receipts = [
            str(_checksum(canonical_json(observation))) for observation in completed
        ]
        if (
            len(response_stages) != len(set(response_stages))
            or len(observation_receipts) != len(set(observation_receipts))
            or len(response_stages) != len(raw_responses)
        ):
            raise ForwardCorpusRejected("official retained response history is ambiguous")
        return response_stages, raw_responses, observation_receipts

    @staticmethod
    def _request_directory(root: Path, race_id: str, request_id: str) -> Path:
        return (
            root
            / "races"
            / hashlib.sha256(race_id.encode()).hexdigest()
            / "official-requests"
            / hashlib.sha256(request_id.encode()).hexdigest()
        )

    def _verify_response_stage(
        self,
        pre: Mapping[str, Any],
        stage: Mapping[str, Any],
    ) -> bytes:
        if set(stage) != {
            "schema_version",
            "race_id",
            "collector_id",
            "session_id",
            "run_id",
            "request_id",
            "source_name",
            "request_url",
            "final_url",
            "http_status",
            "content_type",
            "source_document_last_modified",
            "request_started_at",
            "response_received_at",
            "observed_at",
            "raw_response_checksum",
        } or stage.get("schema_version") != RESPONSE_STAGE_SCHEMA:
            raise ForwardCorpusRejected("official response-stage envelope is invalid")
        if stage.get("race_id") != pre.get("race_id"):
            raise ForwardCorpusRejected("official response-stage race identity mismatch")
        for field in ("collector_id", "session_id", "run_id", "request_id"):
            _bounded_id(stage.get(field), field)
        if stage.get("source_name") != OFFICIAL_RESULT_SOURCE:
            raise ForwardCorpusRejected("official result source name is not canonical")
        _source_url(
            stage.get("request_url"),
            "official request URL",
            allow_official_result_query=True,
        )
        _source_url(
            stage.get("final_url"),
            "official final URL",
            allow_official_result_query=True,
        )
        if type(stage.get("http_status")) is not int or not 200 <= stage["http_status"] < 300:
            raise ForwardCorpusRejected("official response HTTP status is unsupported")
        content_type = _known_text(stage.get("content_type"), "official content type")
        media_type, separator, parameters = content_type.partition(";")
        if media_type.strip().casefold() != "text/html" or (
            separator and parameters.strip().casefold().replace(" ", "") != "charset=utf-8"
        ):
            raise ForwardCorpusRejected("official response content type is unsupported")
        if stage.get("source_document_last_modified") is not None:
            _known_text(stage["source_document_last_modified"], "source document Last-Modified")
        started = _timestamp(stage.get("request_started_at"), "request started")[0]
        received = _timestamp(stage.get("response_received_at"), "response received")[0]
        observed = _timestamp(stage.get("observed_at"), "result observed")[0]
        if not started <= received <= observed:
            raise ForwardCorpusRejected("official response timestamps are unordered")
        return self._read_artifact(stage.get("raw_response_checksum"), "raw response")

    def _verify_observation(
        self,
        pre: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> bytes:
        expected_keys = {
            "schema_version",
            "race_id",
            "collector_id",
            "session_id",
            "run_id",
            "request_id",
            "source_name",
            "request_url",
            "final_url",
            "http_status",
            "content_type",
            "source_document_last_modified",
            "request_started_at",
            "response_received_at",
            "observed_at",
            "raw_response_checksum",
            "normalized_result_checksum",
            "runner_set_hash",
            "parser_hash",
            "schema_hash",
            "implementation_hash",
        }
        if set(observation) != expected_keys or observation.get("schema_version") != (
            "official-result-observation-v1"
        ):
            raise ForwardCorpusRejected("official observation envelope is invalid")
        if observation.get("race_id") != pre["race_id"]:
            raise ForwardCorpusRejected("official observation race identity mismatch")
        for field in ("collector_id", "session_id", "run_id", "request_id"):
            _bounded_id(observation.get(field), field)
        if observation.get("source_name") != OFFICIAL_RESULT_SOURCE:
            raise ForwardCorpusRejected("official result source name is not canonical")
        _source_url(
            observation.get("request_url"),
            "official request URL",
            allow_official_result_query=True,
        )
        _source_url(
            observation.get("final_url"),
            "official final URL",
            allow_official_result_query=True,
        )
        if type(observation.get("http_status")) is not int or not (
            200 <= observation["http_status"] < 300
        ):
            raise ForwardCorpusRejected("official response HTTP status is unsupported")
        content_type = _known_text(observation.get("content_type"), "official content type")
        media_type, separator, parameters = content_type.partition(";")
        if media_type.strip().casefold() != "text/html":
            raise ForwardCorpusRejected("official response content type is unsupported")
        if separator:
            parameter = parameters.strip().casefold().replace(" ", "")
            if parameter != "charset=utf-8":
                raise ForwardCorpusRejected("official response charset is unsupported")
        last_modified = observation.get("source_document_last_modified")
        if last_modified is not None:
            _known_text(last_modified, "source document Last-Modified")
        started = _timestamp(observation.get("request_started_at"), "request started")[0]
        received = _timestamp(observation.get("response_received_at"), "response received")[0]
        observed = _timestamp(observation.get("observed_at"), "result observed")[0]
        if not started <= received <= observed:
            raise ForwardCorpusRejected("official response timestamps are unordered")
        raw = self._read_artifact(observation.get("raw_response_checksum"), "raw response")
        normalized = self._read_artifact(
            observation.get("normalized_result_checksum"), "normalized result"
        )
        source_capture = _canonical_object(
            self._read_artifact(pre["source_capture_checksum"], "source capture"),
            "source capture",
        )
        rebuilt = _normalize_official_result(
            raw,
            race_id=pre["race_id"],
            frozen_runners=source_capture["runners"],
        )
        parser_hash, schema_hash, implementation_hash = _normalization_identity()
        runner_set_hash = str(_checksum(canonical_json(source_capture["runners"])))
        if (
            rebuilt != normalized
            or str(_checksum(rebuilt)) != observation["normalized_result_checksum"]
            or observation["runner_set_hash"] != runner_set_hash
            or observation["parser_hash"] != parser_hash
            or observation["schema_hash"] != schema_hash
            or observation["implementation_hash"] != implementation_hash
        ):
            raise ForwardCorpusRejected("official result reconstruction or code identity drift")
        return normalized

    def capture_result(
        self,
        *,
        race_id: str,
        collector_id: str,
        session_id: str,
        run_id: str,
        request_id: str,
        request_url: str,
        transport: Callable[[str], Mapping[str, Any]],
        source_name: str = OFFICIAL_RESULT_SOURCE,
    ) -> dict[str, Any]:
        """Fetch and retain one official response using collector-owned timestamps."""
        race_id = _bounded_id(race_id, "race_id")
        pre = self._load_receipt(race_id, "prejump")
        if pre is None:
            raise ForwardCorpusRejected("pre-jump evidence must exist before result capture")
        if source_name != OFFICIAL_RESULT_SOURCE:
            raise ForwardCorpusRejected("official result source name is not canonical")
        for value, name in (
            (collector_id, "collector_id"),
            (session_id, "session_id"),
            (run_id, "run_id"),
            (request_id, "request_id"),
            (source_name, "source_name"),
        ):
            _bounded_id(value, name)
        request_url = _source_url(
            request_url,
            "official request URL",
            allow_official_result_query=True,
        )
        request_directory = self._request_directory(self.root, race_id, request_id)
        observation_path = request_directory / "observation.json"
        response_stage_path = request_directory / "response-stage.json"
        if observation_path.exists():
            observation = _canonical_object(
                observation_path.read_bytes(), "observation receipt"
            )
            immutable = {
                "collector_id": collector_id,
                "session_id": session_id,
                "run_id": run_id,
                "request_id": request_id,
                "request_url": request_url,
                "source_name": source_name,
            }
            if any(observation.get(key) != value for key, value in immutable.items()):
                raise ForwardCorpusRejected("request identity replay conflicts with immutable input")
            self._verify_observation(pre, observation)
            stage = self._load_request_receipt(response_stage_path, "response-stage")
            self._verify_observation_stage_binding(pre, observation, stage)
            self._refresh_stability(pre)
            return observation
        stage = self._load_request_receipt(response_stage_path, "response-stage")
        if stage is None:
            self._recover_conflict(pre)
            if self._load_receipt(race_id, "closure") is not None:
                raise ForwardCorpusRejected("new post-closure observation is forbidden")
            if self._load_receipt(race_id, "conflict") is not None:
                raise ForwardCorpusRejected("result identity already changed before closure")
            if not callable(transport):
                raise ForwardCorpusRejected("official transport must be callable")
            started = self._trusted_now("request started")
            response = transport(request_url)
            received = self._trusted_now("response received")
            observed = self._trusted_now("result observed")
            if type(response) is not dict:
                raise ForwardCorpusRejected("official transport response must be a mapping")
            allowed = {
                "body", "status_code", "content_type", "final_url",
                "source_document_last_modified",
            }
            if set(response) - allowed or not {
                "body", "status_code", "content_type", "final_url"
            } <= set(response):
                raise ForwardCorpusRejected("official transport response envelope is invalid")
            body = response["body"]
            if type(body) is not bytes or not body:
                raise ForwardCorpusRejected("immutable official-result response bytes are required")
            raw_checksum = self.artifacts.put(
                body, media_type="application/octet-stream"
            ).checksum
            stage = {
                "schema_version": RESPONSE_STAGE_SCHEMA,
                "race_id": race_id,
                "collector_id": collector_id,
                "session_id": session_id,
                "run_id": run_id,
                "request_id": request_id,
                "source_name": source_name,
                "request_url": request_url,
                "final_url": _source_url(
                    response["final_url"],
                    "official final URL",
                    allow_official_result_query=True,
                ),
                "http_status": response["status_code"],
                "content_type": _known_text(response["content_type"], "official content type"),
                "source_document_last_modified": response.get("source_document_last_modified"),
                "request_started_at": started,
                "response_received_at": received,
                "observed_at": observed,
                "raw_response_checksum": str(raw_checksum),
            }
            self._verify_response_stage(pre, stage)
            stage_bytes = canonical_json(stage)
            self.artifacts.put(stage_bytes, media_type="application/json")
            self._publish_once(response_stage_path, stage_bytes)
        immutable = {
            "collector_id": collector_id, "session_id": session_id, "run_id": run_id,
            "request_id": request_id, "request_url": request_url, "source_name": source_name,
        }
        if any(stage.get(key) != value for key, value in immutable.items()):
            raise ForwardCorpusRejected("request identity replay conflicts with immutable input")
        body = self._verify_response_stage(pre, stage)
        source_capture = _canonical_object(
            self._read_artifact(pre["source_capture_checksum"], "source capture"),
            "source capture",
        )
        normalized = _normalize_official_result(
            body,
            race_id=race_id,
            frozen_runners=source_capture["runners"],
        )
        self._recover_conflict(pre)
        if self._load_receipt(race_id, "closure") is not None:
            raise ForwardCorpusRejected("new post-closure observation is forbidden")
        if self._load_receipt(race_id, "conflict") is not None:
            raise ForwardCorpusRejected("result identity already changed before closure")
        normalized_checksum = self.artifacts.put(
            normalized, media_type="application/json"
        ).checksum
        parser_hash, schema_hash, implementation_hash = _normalization_identity()
        observation = {
            "schema_version": "official-result-observation-v1",
            "race_id": race_id,
            "collector_id": collector_id,
            "session_id": session_id,
            "run_id": run_id,
            "request_id": request_id,
            "source_name": source_name,
            "request_url": request_url,
            "final_url": stage["final_url"],
            "http_status": stage["http_status"],
            "content_type": stage["content_type"],
            "source_document_last_modified": stage["source_document_last_modified"],
            "request_started_at": stage["request_started_at"],
            "response_received_at": stage["response_received_at"],
            "observed_at": stage["observed_at"],
            "raw_response_checksum": stage["raw_response_checksum"],
            "normalized_result_checksum": str(normalized_checksum),
            "runner_set_hash": str(_checksum(canonical_json(source_capture["runners"]))),
            "parser_hash": parser_hash,
            "schema_hash": schema_hash,
            "implementation_hash": implementation_hash,
        }
        self._verify_observation(pre, observation)
        observation_bytes = canonical_json(observation)
        self.artifacts.put(observation_bytes, media_type="application/json")
        self._publish_once(observation_path, observation_bytes)
        self._refresh_stability(pre)
        return observation

    @staticmethod
    def _load_request_receipt(path: Path, name: str) -> dict[str, Any] | None:
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return None
        if path.is_symlink():
            raise ForwardCorpusRejected(f"{name} receipt must not be a symlink")
        return _canonical_object(content, f"{name} receipt")

    def _verify_observation_stage_binding(
        self,
        pre: Mapping[str, Any],
        observation: Mapping[str, Any],
        stage: Mapping[str, Any] | None,
    ) -> None:
        if stage is None:
            raise ForwardCorpusRejected("official response-stage receipt is missing")
        self._verify_response_stage(pre, stage)
        shared = set(stage) - {"schema_version"}
        if any(observation.get(field) != stage[field] for field in shared):
            raise ForwardCorpusRejected("official observation response-stage binding disagrees")

    def _trusted_now(self, name: str) -> str:
        value = self._clock()
        try:
            if type(value) is not datetime:
                raise TypeError
            require_aware(value, name)
        except (TypeError, ValueError) as error:
            raise ForwardCorpusRejected(f"collector {name} timestamp is invalid") from error
        return value.isoformat(timespec="microseconds")

    def _refresh_stability(self, pre: Mapping[str, Any]) -> None:
        observations = self._official_receipts(pre["race_id"])
        self._observation_inventory(pre, observations)
        if not observations:
            return
        identities = {
            (
                item["normalized_result_checksum"],
                item["runner_set_hash"],
                item["parser_hash"],
                item["schema_hash"],
                item["implementation_hash"],
                item["source_name"],
            )
            for item in observations
        }
        if len(identities) > 1:
            conflict = {
                "schema_version": "official-result-conflict-v1",
                "race_id": pre["race_id"],
                "observation_checksums": sorted(
                    str(_checksum(canonical_json(item))) for item in observations
                ),
                "state": "RESULT_CHANGED_BEFORE_CLOSURE",
            }
            self._publish_once(
                self._receipt_path(pre["race_id"], "conflict"), canonical_json(conflict)
            )
            return
        jump = _timestamp(pre["scheduled_jump_at"], "scheduled jump")[0]
        eligible = [
            item
            for item in observations
            if _timestamp(item["observed_at"], "result observed")[0] >= jump + timedelta(minutes=5)
        ]
        eligible.sort(key=lambda item: _timestamp(item["observed_at"], "result observed")[0])
        pair = next(
            (
                (first, second)
                for index, first in enumerate(eligible)
                for second in eligible[index + 1 :]
                if _timestamp(second["observed_at"], "result observed")[0]
                - _timestamp(first["observed_at"], "result observed")[0]
                >= timedelta(minutes=15)
            ),
            None,
        )
        if pair is None:
            return
        receipt = {
            "schema_version": "official-result-stability-v1",
            "race_id": pre["race_id"],
            "first_observation_checksum": str(_checksum(canonical_json(pair[0]))),
            "second_observation_checksum": str(_checksum(canonical_json(pair[1]))),
            "normalized_result_checksum": pair[0]["normalized_result_checksum"],
            "confirmed_at": pair[1]["observed_at"],
        }
        self.artifacts.put(canonical_json(receipt), media_type="application/json")
        self._publish_once(
            self._receipt_path(pre["race_id"], "stability"), canonical_json(receipt)
        )

    def _recover_conflict(self, pre: Mapping[str, Any]) -> None:
        observations = self._official_receipts(pre["race_id"])
        self._observation_inventory(pre, observations)
        if len({self._observation_identity(item) for item in observations}) <= 1:
            return
        conflict = {
            "schema_version": "official-result-conflict-v1",
            "race_id": pre["race_id"],
            "observation_checksums": sorted(
                str(_checksum(canonical_json(item))) for item in observations
            ),
            "state": "RESULT_CHANGED_BEFORE_CLOSURE",
        }
        self._publish_once(
            self._receipt_path(pre["race_id"], "conflict"), canonical_json(conflict)
        )

    @staticmethod
    def _observation_identity(observation: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            observation["normalized_result_checksum"],
            observation["runner_set_hash"],
            observation["parser_hash"],
            observation["schema_hash"],
            observation["implementation_hash"],
            observation["source_name"],
        )

    def _validated_state(
        self, pre: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, bool]:
        observations = self._official_receipts(pre["race_id"])
        self._observation_inventory(pre, observations)
        conflict_present = len({self._observation_identity(item) for item in observations}) > 1
        conflict = self._load_receipt(pre["race_id"], "conflict")
        checksums = sorted(str(_checksum(canonical_json(item))) for item in observations)
        if conflict is not None and conflict != {
            "schema_version": "official-result-conflict-v1",
            "race_id": pre["race_id"],
            "observation_checksums": checksums,
            "state": "RESULT_CHANGED_BEFORE_CLOSURE",
        }:
            raise ForwardCorpusRejected("official conflict receipt binding is invalid")
        if conflict is not None and not conflict_present:
            raise ForwardCorpusRejected("official conflict receipt is not terminal")
        stability = self._load_receipt(pre["race_id"], "stability")
        if stability is not None:
            if set(stability) != {
                "schema_version", "race_id", "first_observation_checksum",
                "second_observation_checksum", "normalized_result_checksum", "confirmed_at",
            } or stability.get("schema_version") != "official-result-stability-v1":
                raise ForwardCorpusRejected("official stability receipt envelope is invalid")
            if stability.get("race_id") != pre["race_id"]:
                raise ForwardCorpusRejected("official stability receipt race identity is invalid")
            by_checksum = {
                str(_checksum(canonical_json(item))): item for item in observations
            }
            try:
                first = by_checksum[stability["first_observation_checksum"]]
                second = by_checksum[stability["second_observation_checksum"]]
            except (KeyError, TypeError) as error:
                raise ForwardCorpusRejected(
                    "stability receipt is not bound to observations"
                ) from error
            if first is second or first["request_id"] == second["request_id"]:
                raise ForwardCorpusRejected("stability observations are not distinct")
            first_at = _timestamp(first["observed_at"], "first observed")[0]
            second_at = _timestamp(second["observed_at"], "second observed")[0]
            jump_at = _timestamp(pre["scheduled_jump_at"], "scheduled jump")[0]
            if (
                self._observation_identity(first) != self._observation_identity(second)
                or first_at < jump_at + timedelta(minutes=5)
                or second_at - first_at < timedelta(minutes=15)
                or stability["normalized_result_checksum"]
                != first["normalized_result_checksum"]
                or _timestamp(stability["confirmed_at"], "stability confirmation")[0]
                != second_at
            ):
                raise ForwardCorpusRejected("official stability receipt binding is invalid")
        return observations, stability, conflict, conflict_present

    def close(self, *, race_id: str) -> dict[str, Any]:
        """Explicitly close a stable race; stability never closes it implicitly."""
        race_id = _bounded_id(race_id, "race_id")
        existing = self._load_receipt(race_id, "closure")
        if existing is not None:
            self._verify_closed_race(existing)
            return existing
        pre = self._load_receipt(race_id, "prejump")
        if pre is None:
            raise ForwardCorpusRejected("result stability must be confirmed before closure")
        observations, stability, _conflict, conflict_present = self._validated_state(pre)
        if conflict_present:
            raise ForwardCorpusRejected("result changed before closure")
        if stability is None:
            raise ForwardCorpusRejected("result stability must be confirmed before closure")
        by_checksum = {
            str(_checksum(canonical_json(item))): item for item in observations
        }
        try:
            first = by_checksum[stability["first_observation_checksum"]]
            second = by_checksum[stability["second_observation_checksum"]]
        except (KeyError, TypeError) as error:
            raise ForwardCorpusRejected("stability receipt is not bound to observations") from error
        self._verify_observation(pre, first)
        self._verify_observation(pre, second)
        closed_at = self._trusted_now("closure")
        if _timestamp(closed_at, "closure")[0] < _timestamp(
            stability["confirmed_at"], "stability confirmation"
        )[0]:
            raise ForwardCorpusRejected("closure predates stability")
        entry = self._race_entry(pre, stability, first, second, observations, closed_at)
        example_bytes = canonical_json(
            {
                "schema_version": "historical-training-example-v1",
                "origin": FORWARD_CORPUS_ORIGIN,
                **entry,
            }
        )
        example_checksum = self.artifacts.put(
            example_bytes, media_type="application/json"
        ).checksum
        entry["artifact_checksum"] = str(example_checksum)
        closure = {
            "schema_version": CLOSURE_RECEIPT_SCHEMA,
            "race_id": race_id,
            "target_bundle_id": pre["target_bundle_id"],
            "feature_schema_checksum": pre["feature_schema_checksum"],
            "missingness_policy_checksum": pre["missingness_policy_checksum"],
            "closed_at": closed_at,
            "race": entry,
        }
        closure_bytes = canonical_json(closure)
        self.artifacts.put(closure_bytes, media_type="application/json")
        self._publish_once(self._receipt_path(race_id, "closure"), closure_bytes)
        return closure

    def _race_entry(
        self,
        pre: Mapping[str, Any],
        stability: Mapping[str, Any],
        first: Mapping[str, Any],
        second: Mapping[str, Any],
        observations: Sequence[Mapping[str, Any]],
        closed_at: str,
    ) -> dict[str, Any]:
        race_id = pre["race_id"]
        response_stages, raw_responses, observation_receipts = (
            self._observation_inventory(pre, observations)
        )
        return {
            "training_example_id": "official-first-"
            + hashlib.sha256(f"{race_id}\0{pre['source_native_race_id']}".encode()).hexdigest(),
            "race_id": race_id,
            "racing_date": pre["racing_date"],
            "source_checksum": pre["source_checksum"],
            "prejump_receipt_checksum": str(_checksum(canonical_json(pre))),
            "source_capture_checksum": pre["source_capture_checksum"],
            "raw_source_checksum": pre["raw_source_checksum"],
            "feature_matrix_checksum": pre["feature_matrix_checksum"],
            "runner_ids": pre["runner_ids"],
            "source_observed_at": pre["source_observed_at"],
            "feature_observed_at": pre["feature_frozen_at"],
            "scheduled_jump_at": pre["scheduled_jump_at"],
            "first_observation_checksum": str(_checksum(canonical_json(first))),
            "second_observation_checksum": str(_checksum(canonical_json(second))),
            "first_raw_response_checksum": first["raw_response_checksum"],
            "second_raw_response_checksum": second["raw_response_checksum"],
            "response_stage_checksums": response_stages,
            "raw_response_checksums": raw_responses,
            "observation_checksums": observation_receipts,
            "normalized_result_checksum": stability["normalized_result_checksum"],
            "stability_checksum": str(_checksum(canonical_json(stability))),
            "stability_confirmed_at": stability["confirmed_at"],
            "closed_at": closed_at,
        }

    def build_package(self) -> ClosedPackage:
        """Build a deterministic package from explicit, verified closures."""
        closures = self._closures()
        if not closures:
            raise ForwardCorpusRejected("no closed races are available")
        for closure in closures:
            self._verify_closed_race(closure)
        bundle_ids = {item["target_bundle_id"] for item in closures}
        schemas = {item["feature_schema_checksum"] for item in closures}
        policies = {item["missingness_policy_checksum"] for item in closures}
        if len(bundle_ids) != 1 or len(schemas) != 1 or len(policies) != 1:
            raise ForwardCorpusRejected("closed races use different feature contracts")
        races = sorted(
            (
                {
                    **item["race"],
                    "closure_receipt_checksum": str(_checksum(canonical_json(item))),
                }
                for item in closures
            ),
            key=lambda race: _identity_key(race["race_id"], "race_id"),
        )
        manifest = {
            "schema_version": "historical-source-manifest-v1",
            "corpus_origin": FORWARD_CORPUS_ORIGIN,
            "target_bundle_id": next(iter(bundle_ids)),
            "feature_schema_checksum": next(iter(schemas)),
            "missingness_policy_checksum": next(iter(policies)),
            "races": races,
        }
        manifest_checksum = _checksum(canonical_json(manifest))
        package_bytes = canonical_json(
            {
                "schema_version": "historical-source-package-v1",
                "manifest_checksum": str(manifest_checksum),
                "manifest": manifest,
            }
        )
        scalar_fields = (
            "source_checksum",
            "prejump_receipt_checksum",
            "closure_receipt_checksum",
            "source_capture_checksum",
            "raw_source_checksum",
            "feature_matrix_checksum",
            "normalized_result_checksum",
            "stability_checksum",
            "artifact_checksum",
        )
        declared = {
            manifest["feature_schema_checksum"],
            manifest["missingness_policy_checksum"],
            *{race[field] for race in races for field in scalar_fields},
            *{
                checksum
                for race in races
                for field in (
                    "response_stage_checksums",
                    "raw_response_checksums",
                    "observation_checksums",
                )
                for checksum in race[field]
            },
        }
        artifacts = {
            checksum: self._read_artifact(checksum, "package artifact")
            for checksum in sorted(declared)
        }
        from .source_admission import admit_historical_source

        admit_historical_source(package_bytes, artifacts=artifacts)
        package_checksum = _checksum(package_bytes)
        self.artifacts.put(package_bytes, media_type="application/json")
        self._publish_once(
            self.root / "packages" / f"{manifest_checksum.hex_digest}.json", package_bytes
        )
        return ClosedPackage(package_bytes, package_checksum, manifest_checksum, artifacts)

    def _verify_closed_race(self, closure: Mapping[str, Any]) -> None:
        pre = self._load_receipt(closure.get("race_id"), "prejump")
        if pre is None:
            raise ForwardCorpusRejected("closure references missing immutable receipt")
        observations, stability, _conflict, conflict_present = self._validated_state(pre)
        if stability is None or conflict_present:
            raise ForwardCorpusRejected("closure references invalid terminal state")
        if set(closure) != {
            "schema_version", "race_id", "target_bundle_id", "feature_schema_checksum",
            "missingness_policy_checksum", "closed_at", "race",
        } or closure.get("schema_version") != CLOSURE_RECEIPT_SCHEMA:
            raise ForwardCorpusRejected("closure receipt envelope is invalid")
        by_checksum = {
            str(_checksum(canonical_json(item))): item for item in observations
        }
        race = closure.get("race")
        if type(race) is not dict:
            raise ForwardCorpusRejected("closure race envelope is invalid")
        try:
            first = by_checksum[race["first_observation_checksum"]]
            second = by_checksum[race["second_observation_checksum"]]
        except (KeyError, TypeError) as error:
            raise ForwardCorpusRejected("closure observation binding is invalid") from error
        self._verify_observation(pre, first)
        self._verify_observation(pre, second)
        expected = self._race_entry(
            pre, stability, first, second, observations, closure["closed_at"]
        )
        artifact_checksum = race.get("artifact_checksum")
        artifact = self._read_artifact(artifact_checksum, "training example")
        if (
            {key: value for key, value in race.items() if key != "artifact_checksum"} != expected
            or artifact
            != canonical_json(
                {
                    "schema_version": "historical-training-example-v1",
                    "origin": FORWARD_CORPUS_ORIGIN,
                    **expected,
                }
            )
        ):
            raise ForwardCorpusRejected("closure or training example binding drift")

    def status(self) -> dict[str, Any]:
        """Reconstruct state from immutable receipts and reparsed raw responses."""
        rows = []
        directory = self.root / "races"
        if directory.exists():
            for race_directory in sorted(path for path in directory.iterdir() if path.is_dir()):
                pre_path = race_directory / "prejump.json"
                if not pre_path.exists():
                    continue
                pre = _canonical_object(pre_path.read_bytes(), "prejump receipt")
                for field in (
                    "raw_source_checksum",
                    "source_checksum",
                    "source_capture_checksum",
                    "feature_schema_checksum",
                    "missingness_policy_checksum",
                    "feature_matrix_checksum",
                ):
                    self._read_artifact(pre[field], field)
                observations = self._official_receipts(pre["race_id"])
                observations, stability, conflict, conflict_present = self._validated_state(pre)
                closure = self._load_receipt(pre["race_id"], "closure")
                state = "RESULT_PENDING"
                if observations:
                    state = "RESULT_FIRST_OBSERVED"
                if stability is not None:
                    state = "RESULT_STABILITY_CONFIRMED"
                if closure is not None:
                    self._verify_closed_race(closure)
                    state = "EXAMPLE_CLOSED"
                if conflict is not None or conflict_present:
                    state = "RESULT_CHANGED_BEFORE_CLOSURE"
                rows.append(
                    {
                        "race_id": pre["race_id"],
                        "state": state,
                        "result_observation_count": len(observations),
                    }
                )
        rows.sort(key=lambda row: _identity_key(row["race_id"], "race_id"))
        return {
            "schema_version": STATUS_SCHEMA,
            "race_count": len(rows),
            "closed_race_count": sum(row["state"] == "EXAMPLE_CLOSED" for row in rows),
            "races": rows,
        }
