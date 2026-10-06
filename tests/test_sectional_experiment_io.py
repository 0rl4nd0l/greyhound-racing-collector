import hashlib
import json
import stat

import pytest

from race_collection.sectional_experiment_io import CheckedReader, PrivateOutput


def test_checked_reader_charges_verified_snapshot_once_and_rejects_other_reference(tmp_path):
    path = tmp_path/'source.json'
    path.write_bytes(b'{"value":1}')
    ref = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    reader = CheckedReader({str(path): ref}, {'max_reads': 1, 'max_bytes': ref['bytes'], 'max_seconds': 30})
    assert reader.json(ref) == {'value': 1}
    assert reader.read(ref) == path.read_bytes()
    assert reader.reads == 1 and reader.bytes == ref['bytes']
    with pytest.raises(ValueError, match='UNBOUND_INPUT'):
        reader.read({**ref, 'sha256': '0'*64})


def test_resource_failure_preserves_consumption_without_returning_input(tmp_path):
    path = tmp_path/'source'
    path.write_bytes(b'123')
    ref = {'path': str(path), 'sha256': hashlib.sha256(b'123').hexdigest(), 'bytes': 3}
    reader = CheckedReader({str(path): ref}, {'max_reads': 1, 'max_bytes': 2, 'max_seconds': 30})
    with pytest.raises(ValueError, match='READ_LIMIT'):
        reader.read(ref)
    assert reader.reads == 1 and reader.bytes == 3 and reader.cache == {}


def test_private_output_limit_keeps_failure_receipt_and_no_success(tmp_path):
    out = PrivateOutput(tmp_path/'output', 65540)
    with pytest.raises(ValueError, match='OUTPUT_LIMIT'):
        out.put('summary.json', {'success': True})
    out.failure('OUTPUT_LIMIT', [], 'race1', ['race2'])
    assert not (out.path/'summary.json').exists()
    failure = out.path/'FAILED.json'
    assert json.loads(failure.read_bytes())['active'] == 'race1'
    assert stat.S_IMODE(failure.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        PrivateOutput(out.path, 100000)
