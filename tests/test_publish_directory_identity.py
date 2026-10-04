"""Directory entry churn must not masquerade as replacement of an opened inode."""
import os
from pathlib import Path
import pytest
from race_collection.synchronous_manual_capture import _atomic_replace_canonical, CaptureOneRejected


def test_strict_publish_guard_still_rejects_concurrent_sibling_creation(tmp_path):
    root=tmp_path/'evidence'; root.mkdir()
    target=root/'state'/'current.json';target.parent.mkdir()
    old=root.stat()
    def sibling_writer():
        (root/'parallel-lane').mkdir()
        new=root.stat()
        assert (new.st_dev,new.st_ino)==(old.st_dev,old.st_ino)
    with pytest.raises(CaptureOneRejected) as rejected:
        _atomic_replace_canonical(target,{'safe':True},evidence_root=root,_pre_replace=sibling_writer)
    assert rejected.value.details['reason']=='publish_root_replaced'
    assert not target.exists()
