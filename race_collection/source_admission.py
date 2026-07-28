"""Pure admission of immutable historical evidence for a native training decision."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any
from urllib.parse import urlsplit

from .domain import ArtifactChecksum, require_aware
from .features import FeatureQuarantine, derive_features
from .model_bundle import SUPPORTED_FEATURE_CONTRACT

SOURCE_MANIFEST_SCHEMA = "historical-source-manifest-v1"
SOURCE_PACKAGE_SCHEMA = "historical-source-package-v1"
ADMITTED_MANIFEST_SCHEMA = "historical-source-admission-v1"
LEGACY_ORIGIN = "legacy-historical-bootstrap-v1"
SYNTHETIC_ORIGIN = "synthetic-validation-fixture-v1"
FORWARD_SEALED_ORIGIN = "forward-sealed-corpus-v1"

_MANIFEST_FIELDS = {
    "schema_version",
    "corpus_origin",
    "target_bundle_id",
    "feature_schema_checksum",
    "missingness_policy_checksum",
    "races",
}
_RACE_FIELDS = {
    "training_example_id",
    "race_id",
    "racing_date",
    "source_checksum",
    "official_result_checksum",
    "feature_matrix_checksum",
    "artifact_checksum",
    "runner_ids",
    "feature_observed_at",
    "scheduled_jump_at",
    "result_published_at",
    "result_observed_at",
}
_FORWARD_RACE_FIELDS = _RACE_FIELDS | {
    "source_capture_checksum",
    "raw_source_checksum",
    "raw_result_checksum",
    "source_observed_at",
}
_RESULT_DERIVED_KEYS = {
    "finish_order",
    "place",
    "result",
    "result_order",
    "winner",
}


class SourceAdmissionRejected(ValueError):
    """Historical source bytes cannot support a truthful training decision."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise SourceAdmissionRejected("source package is not exact JSON") from error


def _checksum(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _object(content: bytes, name: str) -> dict[str, Any]:
    if type(content) is not bytes:
        raise SourceAdmissionRejected(f"{name} must be exact bytes")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceAdmissionRejected(f"{name} is not valid JSON") from error
    if type(value) is not dict or content != _canonical(value):
        raise SourceAdmissionRejected(f"{name} must be one canonical JSON object")
    return value


def _known_text(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value.casefold() == "unknown"
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise SourceAdmissionRejected(f"{name} is missing or ambiguous")
    return value


def _identity_key(value: Any, name: str) -> str:
    identity = _known_text(value, name)
    normalized = unicodedata.normalize("NFKC", " ".join(identity.split())).casefold()
    if not normalized:
        raise SourceAdmissionRejected(f"{name} is missing or ambiguous")
    return normalized


def _canonical_source_url(value: Any, name: str) -> str:
    url = _known_text(value, name)
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise SourceAdmissionRejected(f"{name} is not a canonical source URL")
    return url


def _aware_timestamp(value: Any, name: str) -> datetime:
    if type(value) is not str:
        raise SourceAdmissionRejected(f"{name} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value)
        require_aware(parsed, name)
    except (TypeError, ValueError) as error:
        raise SourceAdmissionRejected(f"{name} timestamp is invalid") from error
    return parsed


def _normalized_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    races = manifest.get("races")
    if type(races) is not list or not races or any(type(race) is not dict for race in races):
        raise SourceAdmissionRejected("source manifest requires race objects")
    expected_fields = (
        _FORWARD_RACE_FIELDS
        if manifest.get("corpus_origin") == FORWARD_SEALED_ORIGIN
        else _RACE_FIELDS
    )
    if any(set(race) != expected_fields for race in races):
        raise SourceAdmissionRejected("source manifest race envelope is invalid")
    try:
        ordered = sorted(races, key=lambda race: _identity_key(race["race_id"], "race_id"))
    except KeyError as error:
        raise SourceAdmissionRejected("source manifest race identity is incomplete") from error
    return {**manifest, "races": ordered}


def _artifact(
    artifacts: Mapping[str, bytes],
    artifact_checksum: Any,
    name: str,
) -> bytes:
    if type(artifact_checksum) is not str:
        raise SourceAdmissionRejected(f"{name} checksum is invalid")
    try:
        ArtifactChecksum(artifact_checksum)
    except ValueError as error:
        raise SourceAdmissionRejected(f"{name} checksum is invalid") from error
    try:
        content = artifacts[artifact_checksum]
    except KeyError as error:
        raise SourceAdmissionRejected(f"{name} checksum is unverifiable") from error
    if type(content) is not bytes or _checksum(content) != artifact_checksum:
        raise SourceAdmissionRejected(f"{name} checksum is unverifiable")
    return content


def _reject_result_derived(value: Any) -> None:
    if type(value) is dict:
        for key, nested in value.items():
            if type(key) is str and key.casefold() in _RESULT_DERIVED_KEYS:
                raise SourceAdmissionRejected("historical source contains a post-result feature")
            _reject_result_derived(nested)
    elif type(value) is list:
        for nested in value:
            _reject_result_derived(nested)


def _require_unique_normalized(values: list[Any], name: str) -> tuple[str, ...]:
    identities = tuple(_known_text(value, name) for value in values)
    normalized = tuple(_identity_key(value, name) for value in identities)
    if len(set(normalized)) != len(normalized):
        raise SourceAdmissionRejected(f"duplicate normalized {name}")
    return identities


def _validate_source(
    source: Mapping[str, Any],
    race: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> Mapping[str, Any]:
    if set(source) != {
        "schema_version",
        "normalization_version",
        "race_id",
        "historical_capture",
        "fields",
    }:
        raise SourceAdmissionRejected("historical source envelope is invalid")
    capture = source.get("historical_capture")
    fields = source.get("fields")
    if (
        type(capture) is not dict
        or set(capture)
        != {
            "source",
            "source_record_id",
            "observed_at",
            "scheduled_jump_at",
            "identity_authority",
            "reconstructed",
        }
        or type(fields) is not dict
        or set(fields) != {"runner_set", "runner_identity", "runner_features"}
    ):
        raise SourceAdmissionRejected("historical source provenance is incomplete")
    if (
        source.get("schema_version") != schema["evidence_schema_version"]
        or source.get("normalization_version") != schema["normalization_version"]
        or source.get("race_id") != race["race_id"]
        or capture.get("observed_at") != race["feature_observed_at"]
        or capture.get("scheduled_jump_at") != race["scheduled_jump_at"]
    ):
        raise SourceAdmissionRejected("historical source identity or timestamps disagree")
    _known_text(capture.get("source"), "historical source")
    _known_text(capture.get("source_record_id"), "historical source record")
    if (
        capture.get("identity_authority") != "source-native"
        or capture.get("reconstructed") is not False
    ):
        raise SourceAdmissionRejected("historical source identity is reconstructed or ambiguous")
    runners = fields.get("runner_set")
    identities = fields.get("runner_identity")
    features = fields.get("runner_features")
    if type(runners) is not list or type(identities) is not dict or type(features) is not dict:
        raise SourceAdmissionRejected("historical source runner identities are incomplete")
    source_runners = _require_unique_normalized(runners, "runner identity")
    if list(source_runners) != sorted(source_runners) or list(source_runners) != race["runner_ids"]:
        raise SourceAdmissionRejected("source and manifest runner identities disagree")
    if (
        set(identities) != set(source_runners)
        or set(features) != set(source_runners)
        or any(identities[runner] != "authoritative" for runner in source_runners)
    ):
        raise SourceAdmissionRejected("historical source runner identities are ambiguous")
    _reject_result_derived(fields)
    expected_features = {field["name"] for field in schema["fields"]}
    if any(
        type(features[runner]) is not dict or set(features[runner]) != expected_features
        for runner in source_runners
    ):
        raise SourceAdmissionRejected("historical source feature envelope disagrees")
    return capture


def _validate_result(
    result: Mapping[str, Any],
    race: Mapping[str, Any],
) -> tuple[str, ...]:
    if (
        set(result)
        != {
            "schema_version",
            "race_id",
            "official",
            "order",
            "published_at",
            "exclusions",
            "provenance",
        }
        or result.get("schema_version") != "official-historical-result-v1"
    ):
        raise SourceAdmissionRejected("official result envelope is invalid")
    provenance = result.get("provenance")
    if (
        type(provenance) is not dict
        or set(provenance)
        != {
            "source",
            "source_record_id",
            "observed_at",
            "identity_authority",
            "reconstructed",
        }
        or result.get("official") is not True
        or result.get("exclusions") != []
        or result.get("race_id") != race["race_id"]
        or result.get("published_at") != race["result_published_at"]
        or provenance.get("observed_at") != race["result_observed_at"]
    ):
        raise SourceAdmissionRejected("official result provenance or identity disagrees")
    _known_text(provenance.get("source"), "official result source")
    _known_text(provenance.get("source_record_id"), "official result source record")
    if (
        provenance.get("identity_authority") != "source-native"
        or provenance.get("reconstructed") is not False
    ):
        raise SourceAdmissionRejected("official result identity is reconstructed or ambiguous")
    order = result.get("order")
    if type(order) is not list:
        raise SourceAdmissionRejected("official result order is incomplete")
    official_order = _require_unique_normalized(order, "official runner identity")
    if len(official_order) < 2 or set(official_order) != set(race["runner_ids"]):
        raise SourceAdmissionRejected("official result and source runner identities disagree")
    return official_order


def _validate_forward_source(
    source: Mapping[str, Any],
    capture: Mapping[str, Any],
    race: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> tuple[str, str, dict[str, str]]:
    if set(source) != {
        "schema_version",
        "normalization_version",
        "race_id",
        "fields",
        "field_provenance",
        "freeze",
    }:
        raise SourceAdmissionRejected("forward sealed evidence envelope is invalid")
    if (
        set(capture)
        != {
            "schema_version",
            "race_id",
            "racing_date",
            "source_name",
            "canonical_source_url",
            "source_native_race_id",
            "meeting_metadata",
            "race_metadata",
            "scheduled_jump_at",
            "source_observed_at",
            "feature_frozen_at",
            "raw_source_checksum",
            "sealed_evidence_checksum",
            "runners",
            "identity_authority",
            "reconstructed",
        }
        or capture.get("schema_version") != "forward-source-capture-v1"
    ):
        raise SourceAdmissionRejected("forward source capture provenance is incomplete")
    if (
        source.get("schema_version") != schema["evidence_schema_version"]
        or source.get("normalization_version") != schema["normalization_version"]
        or source.get("race_id") != race["race_id"]
        or capture.get("race_id") != race["race_id"]
        or capture.get("racing_date") != race["racing_date"]
        or capture.get("scheduled_jump_at") != race["scheduled_jump_at"]
        or capture.get("source_observed_at") != race["source_observed_at"]
        or capture.get("feature_frozen_at") != race["feature_observed_at"]
        or capture.get("raw_source_checksum") != race["raw_source_checksum"]
        or capture.get("sealed_evidence_checksum") != race["source_checksum"]
    ):
        raise SourceAdmissionRejected("forward source identity, hashes, or timestamps disagree")
    _known_text(capture.get("source_name"), "forward source")
    source_url = _canonical_source_url(capture.get("canonical_source_url"), "forward source URL")
    source_native_race_id = _known_text(
        capture.get("source_native_race_id"), "source-native race identity"
    )
    if (
        type(capture.get("meeting_metadata")) is not dict
        or not capture["meeting_metadata"]
        or type(capture.get("race_metadata")) is not dict
        or not capture["race_metadata"]
        or capture.get("identity_authority") != "source-native"
        or capture.get("reconstructed") is not False
    ):
        raise SourceAdmissionRejected("forward source metadata or identity is ambiguous")
    _reject_result_derived(capture["meeting_metadata"])
    _reject_result_derived(capture["race_metadata"])

    runners = capture.get("runners")
    if type(runners) is not list or len(runners) < 2:
        raise SourceAdmissionRejected("source-native runner identities are incomplete")
    if any(
        type(item) is not dict or set(item) != {"source_native_runner_id", "name"}
        for item in runners
    ):
        raise SourceAdmissionRejected("source-native runner identity envelope is invalid")
    runner_ids = _require_unique_normalized(
        [item["source_native_runner_id"] for item in runners],
        "source-native runner identity",
    )
    if list(runner_ids) != sorted(runner_ids) or list(runner_ids) != race["runner_ids"]:
        raise SourceAdmissionRejected("source and manifest runner identities disagree")
    runner_names = {}
    for item in runners:
        _known_text(item["name"], "runner name")
        runner_names[item["source_native_runner_id"]] = item["name"]

    fields = source.get("fields")
    provenance = source.get("field_provenance")
    freeze = source.get("freeze")
    if (
        type(fields) is not dict
        or type(provenance) is not list
        or not provenance
        or type(freeze) is not dict
        or set(freeze) != {"at", "authority", "odds_checksum"}
    ):
        raise SourceAdmissionRejected("forward feature-freeze provenance is incomplete")
    try:
        _known_text(freeze["authority"], "sealed evidence freeze authority")
        ArtifactChecksum(freeze["odds_checksum"])
    except (KeyError, ValueError) as error:
        raise SourceAdmissionRejected("forward feature-freeze provenance is incomplete") from error
    if _aware_timestamp(freeze.get("at"), "sealed evidence freeze") != _aware_timestamp(
        race["feature_observed_at"], "feature observed_at"
    ):
        raise SourceAdmissionRejected("forward feature-freeze provenance is incomplete")
    required_bindings = {"runner_set", "runner_identity", "runner_features"}
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
            raise SourceAdmissionRejected("forward feature source binding is invalid")
        try:
            ArtifactChecksum(item["artifact_checksum"])
        except ValueError as error:
            raise SourceAdmissionRejected(
                "forward feature source binding checksum is invalid"
            ) from error
        _known_text(item.get("field"), "forward feature source binding field")
        _known_text(item.get("authority"), "forward feature source binding authority")
        _known_text(item.get("source"), "forward feature source binding source")
        if type(item.get("critical")) is not bool:
            raise SourceAdmissionRejected("forward feature source binding criticality is invalid")
        if (
            item.get("field") in required_bindings
            and item.get("artifact_checksum") == race["raw_source_checksum"]
            and item.get("source") == capture["source_name"]
            and item.get("value") == fields.get(item["field"])
        ):
            bound_fields.add(item["field"])
    if bound_fields != required_bindings:
        raise SourceAdmissionRejected(
            "forward features are not bound to preserved raw source bytes"
        )
    source_runners = fields.get("runner_set")
    identities = fields.get("runner_identity")
    features = fields.get("runner_features")
    if (
        source_runners != list(runner_ids)
        or type(identities) is not dict
        or set(identities) != set(runner_ids)
        or any(identities[runner] != "authoritative" for runner in runner_ids)
        or type(features) is not dict
        or set(features) != set(runner_ids)
    ):
        raise SourceAdmissionRejected("forward sealed runner identities disagree")
    expected_features = {field["name"] for field in schema["fields"]}
    if any(
        type(features[runner]) is not dict or set(features[runner]) != expected_features
        for runner in runner_ids
    ):
        raise SourceAdmissionRejected("forward sealed feature envelope disagrees")
    _reject_result_derived(source)
    return source_url, source_native_race_id, runner_names


def _validate_forward_result(
    result: Mapping[str, Any],
    race: Mapping[str, Any],
    *,
    expected_source_native_race_id: str,
    expected_runner_names: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, str]]:
    if (
        set(result)
        != {
            "schema_version",
            "race_id",
            "official",
            "order",
            "published_at",
            "exclusions",
            "runner_names",
            "provenance",
        }
        or result.get("schema_version") != "official-forward-result-v1"
    ):
        raise SourceAdmissionRejected("forward official result envelope is invalid")
    provenance = result.get("provenance")
    if (
        type(provenance) is not dict
        or set(provenance)
        != {
            "source",
            "canonical_source_url",
            "source_native_race_id",
            "observed_at",
            "publication_timestamp_status",
            "raw_result_checksum",
            "identity_authority",
            "reconstructed",
        }
        or result.get("official") is not True
        or result.get("exclusions") != []
        or result.get("race_id") != race["race_id"]
        or result.get("published_at") != race["result_published_at"]
        or provenance.get("observed_at") != race["result_observed_at"]
        or provenance.get("raw_result_checksum") != race["raw_result_checksum"]
        or provenance.get("source_native_race_id") != expected_source_native_race_id
        or provenance.get("publication_timestamp_status") != "source-declared"
        or provenance.get("identity_authority") != "source-native"
        or provenance.get("reconstructed") is not False
    ):
        raise SourceAdmissionRejected("forward official result provenance disagrees")
    _known_text(provenance.get("source"), "forward official result source")
    source_url = _canonical_source_url(
        provenance.get("canonical_source_url"), "forward official result source URL"
    )
    order = result.get("order")
    names = result.get("runner_names")
    if type(order) is not list or type(names) is not dict:
        raise SourceAdmissionRejected("forward official result identities are incomplete")
    official_order = _require_unique_normalized(order, "official runner identity")
    if (
        len(official_order) < 2
        or set(official_order) != set(race["runner_ids"])
        or set(names) != set(race["runner_ids"])
        or names != expected_runner_names
    ):
        raise SourceAdmissionRejected(
            "forward official result and source runner identities disagree"
        )
    for name in names.values():
        _known_text(name, "official runner name")
    return official_order, (source_url, expected_source_native_race_id)


def _training_example_document(
    race: Mapping[str, Any],
    *,
    origin: str,
    official_order: tuple[str, ...],
) -> dict[str, Any]:
    document = {
        "schema_version": "historical-training-example-v1",
        "origin": origin,
        "forward_sealed": origin == FORWARD_SEALED_ORIGIN,
        "promotion_evidence_eligible": False,
        "training_example_id": race["training_example_id"],
        "race_id": race["race_id"],
        "racing_date": race["racing_date"],
        "source_checksum": race["source_checksum"],
        "official_result_checksum": race["official_result_checksum"],
        "feature_matrix_checksum": race["feature_matrix_checksum"],
        "runner_ids": race["runner_ids"],
        "official_order": list(official_order),
        "feature_observed_at": race["feature_observed_at"],
        "scheduled_jump_at": race["scheduled_jump_at"],
        "result_published_at": race["result_published_at"],
        "result_observed_at": race["result_observed_at"],
    }
    if origin == FORWARD_SEALED_ORIGIN:
        document.update(
            {
                "source_capture_checksum": race["source_capture_checksum"],
                "raw_source_checksum": race["raw_source_checksum"],
                "raw_result_checksum": race["raw_result_checksum"],
                "source_observed_at": race["source_observed_at"],
            }
        )
    return document


def admit_historical_source(
    package_bytes: bytes,
    *,
    artifacts: Mapping[str, bytes],
) -> bytes:
    """Validate supplied bytes and return one deterministic admitted manifest.

    This function performs no IO or writes. A synthetic package can exercise the
    validator but is always returned as validation-only evidence.
    """

    package = _object(package_bytes, "historical source package")
    if set(package) != {"schema_version", "manifest_checksum", "manifest"}:
        raise SourceAdmissionRejected("historical source package envelope is invalid")
    if package.get("schema_version") != SOURCE_PACKAGE_SCHEMA:
        raise SourceAdmissionRejected("historical source package schema is unsupported")
    manifest = package.get("manifest")
    if type(manifest) is not dict or set(manifest) != _MANIFEST_FIELDS:
        raise SourceAdmissionRejected("historical source manifest envelope is invalid")
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise SourceAdmissionRejected("historical source manifest schema is unsupported")
    normalized_manifest = _normalized_manifest(manifest)
    manifest_checksum = package.get("manifest_checksum")
    if type(manifest_checksum) is not str:
        raise SourceAdmissionRejected("source manifest checksum is invalid")
    try:
        ArtifactChecksum(manifest_checksum)
    except ValueError as error:
        raise SourceAdmissionRejected("source manifest checksum is invalid") from error
    if _checksum(_canonical(normalized_manifest)) != manifest_checksum:
        raise SourceAdmissionRejected("source manifest checksum is unverifiable")

    origin = manifest.get("corpus_origin")
    if origin not in {LEGACY_ORIGIN, SYNTHETIC_ORIGIN, FORWARD_SEALED_ORIGIN}:
        raise SourceAdmissionRejected("historical corpus origin is unsupported or untruthful")
    target_bundle_id = _known_text(manifest.get("target_bundle_id"), "target bundle identity")
    races = normalized_manifest["races"]
    race_ids = [_known_text(race["race_id"], "race_id") for race in races]
    race_keys = [_identity_key(race_id, "race_id") for race_id in race_ids]
    if len(set(race_keys)) != len(race_keys):
        raise SourceAdmissionRejected("duplicate normalized race identity")
    example_ids = [
        _identity_key(race["training_example_id"], "training example identity") for race in races
    ]
    if len(set(example_ids)) != len(example_ids):
        raise SourceAdmissionRejected("duplicate normalized training example identity")

    declared_checksums = [
        manifest["feature_schema_checksum"],
        manifest["missingness_policy_checksum"],
        *[
            race[field]
            for race in races
            for field in (
                "source_checksum",
                "official_result_checksum",
                "feature_matrix_checksum",
                "artifact_checksum",
                *(
                    (
                        "source_capture_checksum",
                        "raw_source_checksum",
                        "raw_result_checksum",
                    )
                    if origin == FORWARD_SEALED_ORIGIN
                    else ()
                ),
            )
        ],
    ]
    if any(type(value) is not str for value in declared_checksums):
        raise SourceAdmissionRejected("declared artifact checksum is invalid")
    unique_identity_checksums = [
        manifest["feature_schema_checksum"],
        manifest["missingness_policy_checksum"],
        *[
            race[field]
            for race in races
            for field in (
                "source_checksum",
                "official_result_checksum",
                "feature_matrix_checksum",
                "artifact_checksum",
                *(("source_capture_checksum",) if origin == FORWARD_SEALED_ORIGIN else ()),
            )
        ],
    ]
    if len(set(unique_identity_checksums)) != len(unique_identity_checksums):
        raise SourceAdmissionRejected("declared artifact identities are duplicated")
    if set(artifacts) != set(declared_checksums):
        raise SourceAdmissionRejected("artifact inventory is missing, duplicate, or ambiguous")

    schema_bytes = _artifact(artifacts, manifest["feature_schema_checksum"], "feature schema")
    missingness_bytes = _artifact(
        artifacts, manifest["missingness_policy_checksum"], "missingness policy"
    )
    schema = _object(schema_bytes, "feature schema")
    missingness = _object(missingness_bytes, "missingness policy")
    if (
        set(schema)
        != {
            "bundle_id",
            "contract_version",
            "evidence_schema_version",
            "normalization_version",
            "fields",
        }
        or schema.get("bundle_id") != target_bundle_id
        or schema.get("contract_version") != SUPPORTED_FEATURE_CONTRACT
        or set(missingness) != {"bundle_id", "feature_contract_version", "imputation"}
        or missingness.get("bundle_id") != target_bundle_id
        or missingness.get("feature_contract_version") != SUPPORTED_FEATURE_CONTRACT
    ):
        raise SourceAdmissionRejected("feature schema or missingness contract is unsupported")
    fields = schema.get("fields")
    if type(fields) is not list or not fields:
        raise SourceAdmissionRejected("unsupported feature width")
    for field in fields:
        if type(field) is not dict or set(field) != (
            {"name", "source_field", "semantics", "encoded_value"}
            if field.get("semantics") == "inapplicable"
            else {"name", "source_field", "semantics"}
        ):
            raise SourceAdmissionRejected("feature schema field envelope is unsupported")

    admitted_races = []
    source_records: set[tuple[str, str]] = set()
    result_records: set[tuple[str, str]] = set()
    for race in races:
        runner_values = race.get("runner_ids")
        if type(runner_values) is not list:
            raise SourceAdmissionRejected("manifest runner identities are incomplete")
        manifest_runners = _require_unique_normalized(runner_values, "runner identity")
        if list(manifest_runners) != sorted(manifest_runners):
            raise SourceAdmissionRejected("manifest runner identities must be sorted")
        try:
            racing_date = date.fromisoformat(race["racing_date"])
        except (TypeError, ValueError) as error:
            raise SourceAdmissionRejected("racing date is invalid") from error
        feature_at = _aware_timestamp(race["feature_observed_at"], "feature observed_at")
        jump_at = _aware_timestamp(race["scheduled_jump_at"], "scheduled jump")
        published_at = _aware_timestamp(race["result_published_at"], "result published_at")
        result_at = _aware_timestamp(race["result_observed_at"], "result observed_at")
        source_at = (
            _aware_timestamp(race["source_observed_at"], "source observed_at")
            if origin == FORWARD_SEALED_ORIGIN
            else feature_at
        )
        if not source_at <= feature_at < jump_at < published_at <= result_at:
            raise SourceAdmissionRejected("historical feature/result temporal order is invalid")
        if racing_date != jump_at.date():
            raise SourceAdmissionRejected("racing date and scheduled jump disagree")

        source_bytes = _artifact(artifacts, race["source_checksum"], "historical source")
        result_bytes = _artifact(artifacts, race["official_result_checksum"], "official result")
        feature_bytes = _artifact(artifacts, race["feature_matrix_checksum"], "feature matrix")
        artifact_bytes = _artifact(
            artifacts, race["artifact_checksum"], "training example artifact"
        )
        source = _object(source_bytes, "historical source")
        result = _object(result_bytes, "official result")
        if origin == FORWARD_SEALED_ORIGIN:
            source_capture_bytes = _artifact(
                artifacts, race["source_capture_checksum"], "forward source capture"
            )
            _artifact(artifacts, race["raw_source_checksum"], "raw forward source")
            _artifact(artifacts, race["raw_result_checksum"], "raw official result")
            source_capture = _object(source_capture_bytes, "forward source capture")
            (
                source_url,
                source_native_race_id,
                source_runner_names,
            ) = _validate_forward_source(source, source_capture, race, schema)
            official_order, result_identity = _validate_forward_result(
                result,
                race,
                expected_source_native_race_id=source_native_race_id,
                expected_runner_names=source_runner_names,
            )
            source_record = (
                _identity_key(source_url, "forward source URL"),
                _identity_key(source_native_race_id, "source-native race identity"),
            )
            result_record = (
                _identity_key(result_identity[0], "forward official result source URL"),
                _identity_key(
                    result_identity[1],
                    "forward official result source-native race identity",
                ),
            )
        else:
            capture = _validate_source(source, race, schema)
            official_order = _validate_result(result, race)
            source_record = (
                _identity_key(capture["source"], "historical source"),
                _identity_key(capture["source_record_id"], "historical source record"),
            )
            result_provenance = result["provenance"]
            result_record = (
                _identity_key(result_provenance["source"], "official result source"),
                _identity_key(
                    result_provenance["source_record_id"],
                    "official result source record",
                ),
            )
        if source_record in source_records or result_record in result_records:
            raise SourceAdmissionRejected("source or result record identity is duplicated")
        source_records.add(source_record)
        result_records.add(result_record)

        try:
            derived = derive_features(
                source_bytes,
                expected_evidence_checksum=ArtifactChecksum(race["source_checksum"]),
                schema_bytes=schema_bytes,
                expected_schema_checksum=ArtifactChecksum(manifest["feature_schema_checksum"]),
                missingness_policy_bytes=missingness_bytes,
                expected_missingness_checksum=ArtifactChecksum(
                    manifest["missingness_policy_checksum"]
                ),
            )
        except (FeatureQuarantine, ValueError) as error:
            raise SourceAdmissionRejected(str(error)) from error
        expected_matrix = _canonical(
            {
                "runner_ids": list(derived.matrix.runner_ids),
                "columns": list(derived.matrix.columns),
                "rows": [list(row) for row in derived.matrix.rows],
            }
        )
        if (
            feature_bytes != expected_matrix
            or _checksum(feature_bytes) != race["feature_matrix_checksum"]
            or any(len(row) != len(derived.matrix.columns) for row in derived.matrix.rows)
        ):
            raise SourceAdmissionRejected("feature matrix width, order, or derived values disagree")
        expected_artifact = _canonical(
            _training_example_document(race, origin=origin, official_order=official_order)
        )
        if artifact_bytes != expected_artifact:
            raise SourceAdmissionRejected("immutable training example artifact disagrees")
        admitted_race = {
            "training_example_id": race["training_example_id"],
            "race_id": race["race_id"],
            "racing_date": race["racing_date"],
            "runner_ids": list(manifest_runners),
            "official_order": list(official_order),
            "source_checksum": race["source_checksum"],
            "official_result_checksum": race["official_result_checksum"],
            "feature_matrix_checksum": race["feature_matrix_checksum"],
            "artifact_checksum": race["artifact_checksum"],
            "feature_observed_at": race["feature_observed_at"],
            "scheduled_jump_at": race["scheduled_jump_at"],
            "result_published_at": race["result_published_at"],
            "result_observed_at": race["result_observed_at"],
        }
        if origin == FORWARD_SEALED_ORIGIN:
            admitted_race.update(
                {
                    "source_capture_checksum": race["source_capture_checksum"],
                    "raw_source_checksum": race["raw_source_checksum"],
                    "raw_result_checksum": race["raw_result_checksum"],
                    "source_observed_at": race["source_observed_at"],
                }
            )
        admitted_races.append(admitted_race)

    admitted = {
        "schema_version": ADMITTED_MANIFEST_SCHEMA,
        "admission_decision": (
            "VALIDATION_ONLY" if origin == SYNTHETIC_ORIGIN else "TRAINING_ADMISSIBLE"
        ),
        "corpus_origin": origin,
        "forward_sealed": origin == FORWARD_SEALED_ORIGIN,
        "promotion_evidence_eligible": False,
        "production_readiness": False,
        "target_bundle_id": target_bundle_id,
        "source_manifest_checksum": manifest_checksum,
        "feature_contract_version": SUPPORTED_FEATURE_CONTRACT,
        "feature_schema_checksum": manifest["feature_schema_checksum"],
        "missingness_policy_checksum": manifest["missingness_policy_checksum"],
        "feature_width": len(fields),
        "missingness_imputation": missingness["imputation"],
        "race_ids": race_ids,
        "races": admitted_races,
    }
    return _canonical(admitted)
