"""Private output projection with native invented retained bodies and validators."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from tests.test_reconcile_comparison_result_identity import case, run

@pytest.fixture
def audit_case(case,monkeypatch):
    from scripts import audit_renewed_engineering_results as audit
    from scripts import reconcile_comparison_result_identity as reconcile
    binding=json.loads(Path(case['authority']['binding']['path']).read_bytes())
    old=json.loads(Path(binding['authority']).read_bytes());plan=json.loads(Path(binding['plan']).read_bytes())
    claim=Path(plan['programme_root'])/binding['plan_sha256']/'attempts'/hashlib.sha256(case['job'].input.race_id.encode()).hexdigest()
    value={**case['authority'],'original_binding':case['authority']['binding'],'jobs':[dict(job_id=case['job'].job_id,race_id=case['job'].input.race_id,
        admission={'path':str(claim/'admission.json')},prior_boundary='SYNTHETIC_ONLY')]}
    monkeypatch.setattr(audit,'load_renewal',lambda *a,**k:(value,binding,plan,old,case['runtime']))
    monkeypatch.setattr(audit,'datetime',reconcile.datetime)
    native_run=audit.subprocess.run
    monkeypatch.setattr(audit.subprocess,'run',lambda command,**k:SimpleNamespace(returncode=0)
        if command[:3]==['git','diff','--quiet'] else native_run(command,**k))
    verify=audit.verify_comparison
    def verification(*a,**k):
        result=verify(*a,**k)
        return {**result,'engineering_evidence':True,'future_race_evidence':False}
    monkeypatch.setattr(audit,'verify_comparison',verification)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=audit.ROOT,text=True).strip()
    return case,lambda:audit.audit(case['authority']['binding'],'SYNTHETIC',source_commit=commit)

def test_retained_audit_detects_valid_evidence_without_closing_queue(audit_case):
    case,audit=audit_case
    report=audit()
    assert report['counts']=={'RETAINED_EVIDENCE_VALIDATED_CLOSURE_NOT_COMMITTED':1}
    assert report['original_evidence_preserved'] and report['provider_requests']==0
    public=json.dumps(report)
    for runner in case['job'].input.ordered_runners:
        assert runner['name'] not in public
    assert 'winner' not in public and 'finish_position' not in public

def test_retained_audit_reverifies_closed_native_result(audit_case):
    case,audit=audit_case
    run(case)
    assert audit()['counts']=={'IDENTITY_VERIFIED_CLOSED':1}

def test_retained_audit_keeps_missing_terminal_position_quarantined(audit_case):
    case,audit=audit_case
    body=Path(case['authority']['body']['path']);body.write_text(body.read_text().replace('3rd','UNKNOWN'))
    path=Path(case['authority']['response']['path']);metadata=json.loads(path.read_bytes())
    metadata.update(sha256=hashlib.sha256(body.read_bytes()).hexdigest(),bytes=body.stat().st_size)
    path.write_text(json.dumps(metadata))
    report=audit()
    assert report['counts']=={'QUARANTINED_NATIVE_VALIDATION_REJECTED':1}
    assert report['original_request_count']==1 and report['provider_requests']==0
