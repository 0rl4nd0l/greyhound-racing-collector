"""Invented providers through real exported worker/collector processes.

All fixture entrypoints install kernel network denial. No runtime code contains
fixture transport, clock overrides or authority bypasses.
"""
import argparse
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from src.predictor.on_demand import canonical_bytes, sha256_file


def setup(root):
    from tests.fixtures.frozen_comparison_case import prepare, use_v2, execute
    case = use_v2(prepare(root), status='AUTHORIZED')
    case['db'].unlink()
    assert execute(case)['phase'] == 'PREDICTION_READY'
    plan = json.loads(case['comparison'].read_bytes())
    campaign = root/'campaign'; campaign.mkdir()
    (campaign/'authorization.json').write_bytes(canonical_bytes({'schema_version':'collector_engineering_campaign_v1',
        'campaign_id':'SYNTHETIC','max_capture_attempts':12,'max_logical_requests':48000,'max_live_seconds':10800}))
    (campaign/'ledger.json').write_bytes(canonical_bytes({'campaign_id':'SYNTHETIC','launches':{},'attempts':[],
                                                       'logical_requests':0,'source_holds':[]}))
    state=root/'private';state.mkdir(mode=0o700)
    mount = subprocess.check_output(['findmnt','-n','-o','TARGET','-T',str(root)],text=True).strip()
    mount_uuid = subprocess.check_output(['findmnt','-n','-o','UUID','-T',str(root)],text=True).strip()
    from utils.sportsbet_access import SportsbetAccess
    source_state=root/'source.json';SportsbetAccess(source_state).initialize(access_basis={'status':'permitted','reference':'SYNTHETIC_NO_NETWORK'})
    authority={'status':'AUTHORIZED_MACHINE_RESULT_RETENTION','plan_sha256':sha256_file(case['comparison']),
        'result_database':str(state/'official-results.sqlite3'),'owner':'SYNTHETIC_SYSTEMD',
        'authority_reference':'SYNTHETIC_ONLY','source_budget_reference':'SYNTHETIC_NO_NETWORK',
        'human_outcome_access':False,'issued_at':plan['activated_at'],
        'runtime':{'schema_version':'comparison_result_runtime_v1','state_root':str(state),
            'storage_mount':{'path':mount,'uuid':mount_uuid},'source_state':str(source_state),
            'prediction_bundles':str(root/'predictions'),'job_store':str(root/'predictions-jobs.db'),
            'campaign_root':str(campaign),'lock_path':str(root/'collector.lock'),
            'expires_at':(datetime.fromisoformat(plan['ends_at'])+timedelta(days=14)).isoformat(),
            'max_races':1000,'max_requests':24000,'max_attempts_per_race':24,'races_per_cycle':8,'max_storage_bytes':32*2**30}}
    ap=root/'result-authority.json';ap.write_bytes(canonical_bytes(authority))
    binding={'plan':str(case['comparison']),'plan_sha256':sha256_file(case['comparison']),
             'authority':str(ap),'authority_sha256':sha256_file(ap)}
    (root/'binding.json').write_bytes(canonical_bytes(binding))
    # Every value here is invented. Public fixture output remains structural.
    (root/'scenario.json').write_bytes(canonical_bytes({'now':(case['jump']+timedelta(minutes=20)).isoformat(),
        'response':'pending','race':case['race_id'],'jump':case['jump'].isoformat()}))
    return {'status':'SYNTHETIC_PREDICTION_READY','forecasts':4,'future_evidence':False}


def patched_clock(root):
    scenario=json.loads((root/'scenario.json').read_bytes())
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):
            value=datetime.fromisoformat(scenario['now'])
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)
    return Clock,scenario


def collector(root, args):
    import scripts.autonomous_official_result_capture as module
    import src.predictor.comparison_result_runtime as runtime
    import race_collection.freshness_campaign as campaign
    from tests.test_append_august_official_results import _html
    from tests.test_predict_market_form_residual import RUNNERS
    clock, scenario = patched_clock(root)
    body=b'<html>Results not yet published</html>'
    if scenario['response'] in ('available','changed','deadheat'):
        body=_html(tuple((box, 'Different Dog' if scenario['response']=='changed' else name,
                    ('1st' if i==1 and scenario['response']=='deadheat' else ('1st','2nd','3rd')[i]))
                    for i,(box,name,*_) in enumerate(RUNNERS)))
    class Raw:
        def read(self,size,**kwargs):return body[:size]
    class Response:
        def __init__(self,url):
            self.url=url;self.raw=Raw();self.status_code=429 if scenario['response']=='denial' else 200
            self.headers={'Content-Type':'text/html; charset=utf-8',**({'Retry-After':'600'} if self.status_code==429 else {})}
        def close(self):pass
    def get(session,url,**kwargs):
        with (root/'synthetic-requests.jsonl').open('a') as stream:
            stream.write(json.dumps({'url':url,'allow_redirects':kwargs['allow_redirects'],'at':scenario['now']})+'\n')
        return Response(url)
    with patch.object(module,'datetime',clock),patch.object(runtime,'datetime',clock),patch.object(campaign,'datetime',clock),patch('requests.Session.get',get):
        return module.main(args)


def queue(root):
    import scripts.run_comparison_result_queue as module
    import src.predictor.comparison_result_runtime as runtime
    clock,_=patched_clock(root)
    original=subprocess.Popen
    def popen(command,**kwargs):
        if command[2:4]==['-m','scripts.autonomous_official_result_capture']:
            command=[sys.executable,'-B','-m','tests.fixtures.persistent_comparison_case','--action','collector','--root',str(root),'--',*command[4:]]
        return original(command,**kwargs)
    with patch.object(module,'datetime',clock),patch.object(runtime,'datetime',clock),patch.object(subprocess,'Popen',popen):
        return module.cycle(root/'binding.json')


if __name__=='__main__':
    from scripts.check_freshness_service import deny_network
    deny_network()
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['setup','queue','collector'],required=True)
    a,remaining=p.parse_known_args()
    if a.action=='collector':raise SystemExit(collector(a.root,remaining[1:] if remaining[:1]==['--'] else remaining))
    print(json.dumps(setup(a.root) if a.action=='setup' else queue(a.root)))
