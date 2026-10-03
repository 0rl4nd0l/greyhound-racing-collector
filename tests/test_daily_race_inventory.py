"""Daily inventory reuse must save discovery without manufacturing fresh inputs."""
from datetime import datetime, timedelta
import hashlib
import json

import pytest
from race_collection.daily_race_inventory import (
    InventoryError, discover_daily_inventory, load_daily_inventory, write_daily_inventory,
)
from scripts import refresh_prejump_upcoming as refresh

NOW = datetime.fromisoformat('2026-10-03T12:00:00+10:00')
RACE = {'url': 'https://www.thedogs.com.au/racing/test/2026-10-03/1/example',
        'date': '2026-10-03', 'venue': 'TEST', 'race_number': 1,
        'race_time': '12:30 PM', 'scheduled_jump_datetime': '2026-10-03T12:30:00+10:00',
        'discovery_time_evidence': {'observed_at': '2026-10-03T11:59:00+10:00'}}


def inventory(tmp_path, **kwargs):
    return write_daily_inventory(tmp_path / 'inventory.json', races=[RACE],
                                source_date='2026-10-03', observed_at=NOW-timedelta(minutes=1), **kwargs)


def test_roundtrip_retains_original_source_time_and_jump(tmp_path):
    ref = inventory(tmp_path)
    value = load_daily_inventory(**ref, source_date='2026-10-03', now=NOW)
    assert value['observed_at'] == (NOW-timedelta(minutes=1)).isoformat()
    assert value['races'] == [RACE]
    with pytest.raises(FileExistsError):
        inventory(tmp_path)


@pytest.mark.parametrize('change,reason', [('stale','STALE'), ('future','FUTURE'),
    ('date','SOURCE_DATE'), ('tamper','HASH'), ('incomplete','INCOMPLETE'), ('count','COUNT')])
def test_invalid_inventory_is_rejected_before_selection(tmp_path, change, reason):
    ref = inventory(tmp_path)
    now = NOW
    day = '2026-10-03'
    if change == 'stale': now += timedelta(minutes=16)
    if change == 'future': now -= timedelta(minutes=2)
    if change == 'date': day = '2026-10-04'
    if change in {'tamper','incomplete','count'}:
        p = tmp_path/'inventory.json'; value=json.loads(p.read_text())
        if change == 'incomplete': value['discovery_failures']=[{'error_type':'HTTPStatusError'}]
        elif change == 'count': value['race_count']=2
        else: value['races'][0]['race_time']='12:31 PM'
        p.chmod(0o600)
        p.write_text(json.dumps(value))
        if change != 'tamper': ref['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(InventoryError, match=reason):
        load_daily_inventory(**ref, source_date=day, now=now)


def test_complete_discovery_required_and_duplicate_identity_forbidden(tmp_path):
    with pytest.raises(InventoryError, match='INCOMPLETE'):
        inventory(tmp_path, discovery_failures=[{'error_type':'RequestGuardStopped'}])
    assert not (tmp_path/'inventory.json').exists()
    with pytest.raises(InventoryError, match='DUPLICATE'):
        write_daily_inventory(tmp_path/'duplicate.json', races=[RACE,RACE],
                              source_date='2026-10-03', observed_at=NOW)


def test_discover_uses_explicit_full_day_not_upcoming_filter(tmp_path):
    calls=[]
    class Browser:
        discovery_failures=[]
        def get_races_for_date(self, day):
            calls.append(day.isoformat()); return [RACE]
        def get_upcoming_races(self, **kwargs):
            pytest.fail('upcoming filtering loses elapsed identities')
    ref=discover_daily_inventory(Browser(), source_date='2026-10-03',
                                 path=tmp_path/'inventory.json', observed_at=NOW)
    assert calls==['2026-10-03']
    assert load_daily_inventory(**ref, source_date='2026-10-03',now=NOW)['race_count']==1


def test_native_refresh_reuses_calendar_but_still_downloads_fresh_selected_inputs(tmp_path,monkeypatch):
    ref=inventory(tmp_path); calls=[]
    class Browser:
        discovery_failures=[]
        def get_upcoming_races(self, **kwargs):
            pytest.fail('valid pinned daily inventory must avoid full-calendar requests')
        def download_race_csv(self,url,**kwargs):
            calls.append((url,kwargs)); return {'success':False,'reason':'test_current_inputs_unavailable'}
    monkeypatch.setattr(refresh,'_browser_type',lambda: Browser)
    monkeypatch.setattr(refresh,'_refresh_browser',lambda *args: Browser())
    args=refresh.build_parser().parse_args(['--upcoming-dir',str(tmp_path/'current'),
        '--current-time',NOW.isoformat(),'--min-minutes','5','--max-minutes','60',
        '--discovery-inventory',ref['path'],'--discovery-inventory-sha256',ref['sha256']])
    report=refresh.refresh_prejump_upcoming(args)
    assert len(calls)==1 and calls[0][0]==RACE['url']
    assert report['discovery_inventory']['observed_at']==(NOW-timedelta(minutes=1)).isoformat()
    assert report['discovery_observed_at'] != report['generated_at']
    assert report['current_index_race_count']==0
    assert report['status']!='SUCCESS'


def test_stale_inventory_fails_before_browser_or_download(tmp_path,monkeypatch):
    ref=inventory(tmp_path)
    monkeypatch.setattr(refresh,'_browser_type',lambda: pytest.fail('must validate before browser setup'))
    args=refresh.build_parser().parse_args(['--upcoming-dir',str(tmp_path/'current'),
        '--current-time',(NOW+timedelta(minutes=16)).isoformat(),
        '--discovery-inventory',ref['path'],'--discovery-inventory-sha256',ref['sha256']])
    with pytest.raises(InventoryError,match='STALE'):
        refresh.refresh_prejump_upcoming(args)


@pytest.mark.parametrize('odds_only',[False,True])
def test_native_daemon_cycle_forwards_pinned_inventory_to_phase(tmp_path,monkeypatch,odds_only):
    from scripts import shadow_autopilot_daemon as daemon
    from tests.test_live_collection_cycle import test_native_entrypoint_accounts_for_nonzero_overhead
    original_parse=daemon.parse_args
    original_command_setattr=monkeypatch.setattr
    seen=[]
    flags=['--discovery-inventory',str(tmp_path/'pin.json'),
           '--discovery-inventory-sha256','a'*64,
           '--discovery-inventory-source-date','2026-07-19',
           '--discovery-inventory-max-age-seconds','900']
    def parse(argv): return original_parse([*argv,*flags])
    original_command_setattr(daemon,'parse_args',parse)
    def wrapping_setattr(target,name,value,*args,**kwargs):
        if target is daemon and name=='run_command':
            operation=value
            def wrapped(*args,**kwargs):
                command=kwargs['command']
                for option,expected in zip(flags[::2],flags[1::2]):
                    got=command[command.index(option)+1]
                    assert float(got)==float(expected) if option.endswith('seconds') else got==expected
                seen.append(command)
                return operation(*args,**kwargs)
            value=wrapped
        return original_command_setattr(target,name,value,*args,**kwargs)
    original_command_setattr(monkeypatch,'setattr',wrapping_setattr)
    test_native_entrypoint_accounts_for_nonzero_overhead(
        tmp_path,monkeypatch,odds_only,2,0.25,1,0,'LIVE_COLLECTION_COMPLETE')
    assert len(seen)==1


@pytest.mark.parametrize('odds_only',[False,True])
def test_native_phase_forwards_inventory_to_actual_refresh_parser(tmp_path,monkeypatch,odds_only):
    from scripts import shadow_autopilot_v1 as autopilot
    ref=inventory(tmp_path)
    flags=['--discovery-inventory',ref['path'],'--discovery-inventory-sha256',ref['sha256'],
           '--discovery-inventory-source-date','2026-10-03','--discovery-inventory-max-age-seconds','900']
    argv=['--evidence-root',str(tmp_path/'evidence'),'--run-id','inventory',
          '--collection-phase','refresh','--skip-shadow-run',*flags]
    if odds_only: argv+=['--skip-primary-refresh','--enable-autonomous-odds-capture']
    args=autopilot.parse_args(argv)
    monkeypatch.setattr(autopilot,'protected_hashes',lambda:{})
    monkeypatch.setattr(autopilot,'scheduled_collector_authority',lambda *a,**k:{'run_id':'owned'})
    class Observed(BaseException): pass
    def step(**kwargs):
        command=kwargs['command']; offset=next(i for i,x in enumerate(command) if x.endswith('refresh_prejump_upcoming.py'))
        child=refresh.build_parser().parse_args(command[offset+1:])
        assert child.discovery_inventory==tmp_path/'inventory.json'
        assert child.discovery_inventory_sha256==ref['sha256']
        assert child.discovery_inventory_source_date=='2026-10-03'
        assert child.discovery_inventory_max_age_seconds==900
        assert child.workers==2
        assert not child.dry_run
        raise Observed
    monkeypatch.setattr(autopilot,'step_command',step)
    with pytest.raises(Observed): autopilot.run_autopilot(args)


def test_idle_age_allowance_cannot_relax_active_refresh_freshness(tmp_path,monkeypatch):
    ref=inventory(tmp_path)
    assert load_daily_inventory(**ref,source_date='2026-10-03',
        now=NOW+timedelta(minutes=20),max_age_seconds=1800)
    monkeypatch.setattr(refresh,'_browser_type',lambda: pytest.fail('no setup'))
    args=refresh.build_parser().parse_args(['--upcoming-dir',str(tmp_path/'current'),
        '--current-time',NOW.isoformat(),'--discovery-inventory',ref['path'],
        '--discovery-inventory-sha256',ref['sha256'],'--discovery-inventory-max-age-seconds','1800'])
    with pytest.raises(InventoryError,match='ACTIVE_MAX_AGE'):
        refresh.refresh_prejump_upcoming(args)


def test_failed_discovery_does_not_publish_cached_fallback(tmp_path):
    class Browser:
        discovery_failures=[{'error_type':'EmptyDiscovery'}]
        def get_races_for_date(self,day): return [RACE]
    with pytest.raises(InventoryError,match='INCOMPLETE'):
        discover_daily_inventory(Browser(),source_date='2026-10-03',
                                 path=tmp_path/'inventory.json',observed_at=NOW)
    assert not (tmp_path/'inventory.json').exists()
