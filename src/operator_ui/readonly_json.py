"""Bounded read of canonical regular JSON evidence, with optional hash pin."""
import hashlib
import json
import os
import stat
from pathlib import Path

def read(path, expected=None, limit=4 * 1024 * 1024):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('unsafe_retained_path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError('retained_file_bound')
        raw = os.read(fd, limit + 1)
        after = os.fstat(fd)
        identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if identity(before) != identity(after) or identity(path.stat()) != identity(after) or len(raw) != before.st_size:
            raise ValueError('retained_file_changed')
    finally:
        os.close(fd)
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('retained_identity_mismatch')
    return json.loads(raw)

