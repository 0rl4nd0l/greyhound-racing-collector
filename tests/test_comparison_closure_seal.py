from datetime import datetime,timezone
import json,hashlib
import pytest
from scripts.seal_comparison_result_closure import seal


def test_closure_copy_is_hash_bound_and_never_decodes_database(tmp_path,monkeypatch):
    import scripts.seal_comparison_result_closure as module
    plan={'status':'AUTHORIZED','ends_at':'2026-11-01T00:00:00+00:00'}
    monkeypatch.setattr(module,'load_plan',lambda *a:(plan,b''))
    database=tmp_path/'source.db';raw=b'SQLite opaque synthetic bytes\xff';database.write_bytes(raw)
    authority={'status':'AUTHORIZED_MACHINE_RESULT_RETENTION','plan_sha256':'a'*64,'owner':'synthetic','authority_reference':'synthetic','human_outcome_access':False,'result_database':str(database)}
    auth=tmp_path/'authority.json';auth.write_text(json.dumps(authority))
    binding=tmp_path/'binding.json';binding.write_text(json.dumps({'plan':'unused','plan_sha256':'a'*64,'authority':str(auth),'authority_sha256':hashlib.sha256(auth.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError,match='not_due'):seal(binding,tmp_path/'early',now=datetime(2026,11,14,tzinfo=timezone.utc))
    result=seal(binding,tmp_path/'closure',now=datetime(2026,11,15,tzinfo=timezone.utc))
    assert result['target_values_decoded'] is False and result['result_database_sha256']==hashlib.sha256(raw).hexdigest()
    assert database.read_bytes()==raw and (tmp_path/'closure/official-results.sqlite3').read_bytes()==raw
    assert (tmp_path/'closure/official-results.sqlite3').stat().st_mode & 0o777==0o400
    with pytest.raises(FileExistsError):seal(binding,tmp_path/'closure',now=datetime(2026,11,15,tzinfo=timezone.utc))
