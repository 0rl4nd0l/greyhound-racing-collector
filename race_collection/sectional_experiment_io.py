"""Finite exact-reference local IO and exclusive private artifacts."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


class CheckedReader:
    def __init__(self, allowed, limits):
        self.allowed = allowed
        self.limits = limits
        self.reads = self.bytes = 0
        self.started = time.monotonic()
        self.cache = {}

    def check(self):
        if time.monotonic()-self.started > self.limits['max_seconds']:
            raise ValueError('READ_DEADLINE')

    def read(self, ref):
        self.check()
        path = Path(ref['path'])
        bound = self.allowed.get(str(path))
        if not bound or ref['sha256'] != bound['sha256'] or path.resolve() != path:
            raise ValueError('UNBOUND_INPUT')
        if str(path) not in self.cache:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as f:
                before = os.fstat(f.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != bound['bytes']:
                    raise ValueError('INPUT_SIZE_OR_TYPE')
                self.reads += 1
                self.bytes += before.st_size
                if self.reads > self.limits['max_reads'] or self.bytes > self.limits['max_bytes']:
                    raise ValueError('READ_LIMIT')
                value = f.read(bound['bytes']+1)
                after = os.fstat(f.fileno())
            key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if key(before) != key(after) or key(after) != key(path.stat()) or len(value) != bound['bytes']:
                raise ValueError('INPUT_CHANGED')
            if hashlib.sha256(value).hexdigest() != ref['sha256']:
                raise ValueError('INPUT_HASH')
            self.cache[str(path)] = value
        value = self.cache[str(path)]
        if ref.get('bytes', len(value)) != len(value):
            raise ValueError('REFERENCE_SIZE')
        self.check()
        return value

    def json(self, ref):
        return json.loads(self.read(ref))


class PrivateOutput:
    def __init__(self, path, maximum):
        self.path = Path(path)
        self.path.mkdir(mode=0o700, exist_ok=False)
        self.maximum = maximum
        self.bytes = 0
        self.files = []

    def put(self, name, value):
        if Path(name).name != name:
            raise ValueError('OUTPUT_NAME')
        payload = encoded(value)
        self.bytes += len(payload)
        if self.bytes + 65536 > self.maximum:
            raise ValueError('OUTPUT_LIMIT')
        path = self.path/name
        with path.open('xb') as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        ref = {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
        self.files.append(ref)
        return ref

    def failure(self, reason, completed, active, unattempted, **details):
        # Reserve accounting remains available after the normal output cap fails.
        payload = encoded({'status': 'FAILED_NO_SUCCESS', 'reason': reason,
            'completed': completed, 'active': active, 'unattempted': unattempted,
            'details': details})
        if len(payload) > 65536:
            raise ValueError('FAILURE_ACCOUNTING_LIMIT')
        with (self.path/'FAILED.json').open('xb') as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
