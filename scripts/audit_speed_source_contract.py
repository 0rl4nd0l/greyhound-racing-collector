"""Offline, outcome-free inspection of timing plumbing; never certifies semantics.

Reads a fixed list of local code/definition files, without importing collectors,
opening race data, or making requests. JSON stdout contains code keys and refs only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


INPUTS = {
    'parser': 'src/collectors/fasttrack_scraper.py',
    'adapter': 'src/collectors/adapters/fasttrack_adapter.py',
    'csv': 'csv_ingestion.py',
    'expert': 'utils/expert_form_metadata.py',
    'definitions': 'docs/research/prediction_source_definition_matrix_20261003.json',
}
MAX_FILE_BYTES = 1_000_000
FIELDS = {'1 SEC', 'TIME', 'WIN', 'BON', 'PIR', 'split1', 'Best 1st Split'}


def read_source(root: Path, relative: str) -> tuple[str, dict]:
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('source_path_outside_checkout')
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('source_file_missing_or_oversize')
    raw = path.read_bytes()
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError('source_file_oversize')
    return raw.decode('utf-8'), {
        'path': relative, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
    }


def function(text: str, name: str) -> ast.FunctionDef:
    matches = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError('source_function_missing_or_ambiguous')
    return matches[0]


def literal_keys_assigned(fn: ast.FunctionDef, variable: str) -> list[str]:
    keys = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == variable:
                    if not isinstance(target.slice, ast.Constant) or not isinstance(target.slice.value, str):
                        raise ValueError('dynamic_runner_key_not_supported')
                    keys.add(target.slice.value)
    return sorted(keys)


def literal_get_keys(fn: ast.FunctionDef, variable: str) -> list[str]:
    keys = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == variable and node.func.attr == 'get':
            if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                raise ValueError('dynamic_adapter_key_not_supported')
            keys.add(node.args[0].value)
    return sorted(keys)


def csv_mappings(text: str) -> dict:
    mappings = {}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'ColumnMapping':
            values = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg in {'source_name', 'target_name', 'aliases'}}
            if values.get('source_name') in FIELDS:
                key = values.pop('source_name')
                if key in mappings:
                    raise ValueError('duplicate_csv_mapping')
                mappings[key] = values
    if set(mappings) != {'1 SEC', 'TIME', 'WIN', 'BON', 'PIR'}:
        raise ValueError('timing_csv_mapping_changed')
    return mappings


def definition_status(text: str) -> list[dict]:
    data = json.loads(text)
    rows = data['fields']
    if len(rows) != len(FIELDS) or {row['field'] for row in rows} != FIELDS:
        raise ValueError('definition_field_set_changed')
    report = []
    for row in rows:
        missing = row.get('missing')
        if not isinstance(missing, list) or not all(isinstance(x, str) for x in missing):
            raise ValueError('definition_missing_requirements_invalid')
        # A hand-edited boolean cannot qualify a source-owned measurement.
        claimed = row.get('source_definition_verified') is True or row.get('qualification') != 'UNQUALIFIED' or not missing
        report.append({'field': row['field'], 'status': 'INDEPENDENT_SOURCE_DEFINITION_REVIEW_REQUIRED' if claimed else 'UNQUALIFIED', 'missing': missing})
    return sorted(report, key=lambda row: row['field'])


def audit(root: Path) -> dict:
    texts, refs = {}, {}
    for role, path in INPUTS.items():
        texts[role], refs[role] = read_source(root, path)
    parsed = function(texts['parser'], '_parse_race')
    adapted = function(texts['adapter'], 'adapt_and_load_race')
    emitted = literal_keys_assigned(parsed, 'dog_data')
    consumed = literal_get_keys(adapted, 'dog_perf')
    expert = function(texts['expert'], '_parse_track_distance_cell')
    expert_keys = sorted({key.value for node in ast.walk(expert) if isinstance(node, ast.Dict) for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)})
    return {
        'schema_version': 'speed_source_contract_audit_v1',
        'status': 'SOURCE_SEMANTICS_NOT_QUALIFIED',
        'scope': 'Static local code contract only; not a current-feed census or runtime dataflow proof.',
        'sources': refs,
        'csv_mappings': csv_mappings(texts['csv']),
        'fasttrack': {
            'parser_runner_literal_assignments': emitted,
            'adapter_runner_literal_gets': consumed,
            'adapter_keys_without_parser_literal_assignment': sorted(set(consumed) - set(emitted)),
            'parser_function_line': parsed.lineno,
            'adapter_function_line': adapted.lineno,
            'timing_meaning_inferred': False,
            'active_collector_usage_established': False,
        },
        'expert_track_distance_metadata_keys': expert_keys,
        'definitions': definition_status(texts['definitions']),
        'current_feed_numeric_coverage': 'NOT_MEASURED',
        'current_feed_qualified_coverage': 'NOT_MEASURED',
        'training_or_feature_use_authorized': False,
        'requests': {'python': 0, 'browser': 0, 'source_operations': 0, 'captures': 0, 'results': 0},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = audit(args.source_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0  # Successful audit execution; the report explicitly denies qualification.


if __name__ == '__main__':
    raise SystemExit(main())
