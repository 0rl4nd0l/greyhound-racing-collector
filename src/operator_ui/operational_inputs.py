"""Read-only binding to one authorized collector session and its sealed inputs.

Missing publications are ordinary unavailable inputs. No collection, history
reads, synthetic index, or study fallback is performed by this adapter.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path


def validate_config(value):
    if not isinstance(value, dict) or set(value) != {'evidence_root', 'retained_root'}:
        raise ValueError('invalid operational input binding')
    for name, raw in value.items():
        if not isinstance(raw, str):
            raise ValueError('invalid operational input path')
        path = Path(raw)
        if not path.is_absolute() or path.resolve() != path:
            raise ValueError('operational input path must be canonical')
        # The collector creates the evidence directory at session preparation.
        # Reject aliases even before its first publication exists.
        for parent in (path, *path.parents):
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError('unsafe operational input path')
        if name == 'retained_root' and not path.is_dir():
            raise ValueError('retained input root unavailable')
    return dict(value)


def retained_binding(root, race, now):
    from src.predictor.retained_inputs import _read, _validate_retention_metadata
    root = Path(root)
    directory = root/'races'/hashlib.sha256(race['race_id'].encode()).hexdigest()/'retention'
    candidates = list(directory.glob('*/bundle'))
    if len(candidates) != 1:
        raise ValueError('retained input missing or ambiguous')
    bundle = candidates[0]
    try:
        raw = _read(bundle, 'manifest.json')
        digest = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        completion = json.loads(_read(bundle, 'completion.json'))
        terminal = json.loads(_read(bundle.parent, 'terminal.json'))
        _validate_retention_metadata(manifest, completion,
            expected_manifest_sha256=digest, race_id=race['race_id'],
            jump=datetime.fromisoformat(race['jump_datetime']), now=now)
        if terminal.get('status') != 'RETAINED' or terminal.get('manifest_sha256') != digest:
            raise ValueError('retained input incomplete')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError('retained input unavailable or invalid') from exc
    # The existing predictor validates every input, receipt, model and feature
    # hash before scoring; this pins the selected manifest into job identity.
    return {'path':str(bundle), 'manifest_sha256':digest}
