"""Approval materialization is offline and separate from installing authority."""
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import sys
import pytest
from scripts.prepare_persistent_comparison import prepare
from scripts.authorize_persistent_comparison import materialize
from src.predictor.on_demand import canonical_bytes


def test_hash_bound_materializer_changes_neither_campaign_nor_source(tmp_path,monkeypatch):
    from tests.fixtures.persistent_comparison_case import setup
    setup(tmp_path)
    source=tmp_path/'source.json';campaign=tmp_path/'campaign'
    roots=tmp_path/'roots.json';roots.write_bytes(canonical_bytes({}))
    start=(datetime.now(timezone.utc)+timedelta(days=14)).replace(hour=1,minute=0,second=0,microsecond=0)
    out=tmp_path/'prepared';root=tmp_path/'future'
    pin='a'*40
    prepare(out,root,start.isoformat(),pin,campaign,source,Path(sys.executable),roots)
    before={str(p):p.read_bytes() for p in campaign.rglob('*') if p.is_file()};source_before=source.read_bytes()
    monkeypatch.setattr('scripts.authorize_persistent_comparison.check_mount',lambda *a:None)
    monkeypatch.setattr('race_collection.persistent_storage.check_mount',lambda *a:None)
    monkeypatch.setattr('subprocess.check_output',lambda args,**kwargs: pin+'\n' if args[1]=='rev-parse' else '')
    approved=tmp_path/'approved'
    manifest_hash=hashlib.sha256((out/'prepared-manifest.json').read_bytes()).hexdigest()
    kw=dict(prepared_manifest_sha256=manifest_hash,approval_reference='SYNTHETIC',allocation_reference='SYNTHETIC',history_reference='SYNTHETIC',result_reference='SYNTHETIC')
    assert materialize(out,approved,**kw)['status']=='APPROVAL_FILES_MATERIALIZED_NOT_DEPLOYED'
    assert before=={str(p):p.read_bytes() for p in campaign.rglob('*') if p.is_file()}
    assert source.read_bytes()==source_before
    from scripts.run_comparison_schedule import load_config
    cfg,_=load_config(approved/'schedule.APPROVED.json')
    assert len(cfg['slots'])==80
    from src.predictor.comparison_result_runtime import load_runtime
    binding=json.loads((approved/'result-binding.APPROVED.json').read_bytes())
    _,_,runtime=load_runtime(binding,now=datetime.now(timezone.utc))
    assert runtime['max_requests']==24000
    (out/'schedule.prepared.json').write_bytes(b'{}')
    with pytest.raises(ValueError,match='hash_changed'):materialize(out,tmp_path/'unauthorized',**kw)
    assert not (tmp_path/'unauthorized').exists()
