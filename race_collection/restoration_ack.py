"""Explicit restoration-only acknowledgement of a known operator UI restart."""
import hashlib
import json
from pathlib import Path
from race_collection.live_freshness_contract import digest


def verify_r3_replacement(reference, output, plan, backup, control):
    def require(value):
        if not value:
            raise ValueError('r3_restoration_ack_invalid')

    path, expected_sha, approval = reference
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha)
    value = json.loads(raw)
    require(value['schema_version'] == 'r3_restoration_ack_v1')
    require(value['authority_reference'] == approval and approval)
    require(value['plan_sha256'] == digest(plan))
    require(value['restoration_sha256'] == hashlib.sha256((output/'restoration.json').read_bytes()).hexdigest())
    require(value['old_pid'] == backup['r3_pid'])
    require(json.loads((output/'failure.json').read_bytes())['reason'] == 'installed_r3_changed')
    require(value['collection_resume_allowed'] is False)
    require(value['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    status = control.show('greyhound-operator-ui-r3.service')
    require(status['ActiveState'] == 'active' and status['SubState'] == 'running')
    require(int(status['MainPID']) > 0 and status['MainPID'] == value['new_pid'])
    require(status['InvocationID'] == value['invocation_id'])
    # /proc field 22 is process start time, guarded against PID reuse.
    fields = Path('/proc', status['MainPID'], 'stat').read_text().rsplit(')', 1)[1].split()
    require(fields[19] == value['process_start_ticks'])
    require(value['unit_sha256'] == backup['hashes']['greyhound-operator-ui-r3.service'])
    require(hashlib.sha256((Path(plan['installed_dir'])/'greyhound-operator-ui-r3.service').read_bytes()).hexdigest() == value['unit_sha256'])
    require(value['pinned_files'])
    for filename, expected in value['pinned_files'].items():
        require(hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected)
    return {'path': str(path), 'sha256': expected_sha, 'old_pid': value['old_pid'], 'new_pid': value['new_pid']}
