"""Read-only mount identity check before creating programme files."""
import os
from pathlib import Path
import subprocess


def check_mount(binding, root):
    mount = Path(binding['path'])
    if (not mount.is_absolute() or mount.resolve() != mount or not os.path.ismount(mount)
            or not root.is_absolute() or root.resolve() != root or not root.is_relative_to(mount)):
        raise ValueError('programme_volume_not_mounted')
    uuid = subprocess.check_output(['findmnt','--noheadings','--output','UUID','--target',str(mount)],
                                   text=True,timeout=5).strip()
    if not uuid or uuid != binding['uuid']:
        raise ValueError('programme_volume_identity_changed')
