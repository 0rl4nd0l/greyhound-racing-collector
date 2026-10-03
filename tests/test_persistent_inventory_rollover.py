"""Native refresh may use yesterday's authorized inventory after midnight."""
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from race_collection.daily_race_inventory import write_daily_inventory
from race_collection.live_freshness_contract import FreshnessContract
from scripts import refresh_prejump_upcoming as refresh


@pytest.mark.parametrize('persistent,pinned,allowed', [(True,True,True),(False,True,False),(True,False,False)])
def test_native_refresh_midnight_requires_authenticated_persistent_inventory(tmp_path,monkeypatch,persistent,pinned,allowed):
    zone = ZoneInfo('Australia/Melbourne')
    now = datetime(2026,10,4,0,5,tzinfo=zone)
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return now.astimezone(tz) if tz else now.replace(tzinfo=None)
    monkeypatch.setattr(refresh,'datetime',Clock)
    args = ['--upcoming-dir',str(tmp_path/'current'),'--current-time',now.isoformat(),
            '--live-freshness-contract',str(tmp_path/'scope.json')]
    if pinned:
        ref = write_daily_inventory(tmp_path/'inventory.json',races=[{
            'date':'2026-10-03','venue':'CAN','race_number':11,'race_time':'12:13 AM',
            'scheduled_jump_datetime':'2026-10-04T00:13:00+10:00',
            'url':'https://www.thedogs.com.au/racing/cannington/2026-10-03/11/example'}],
            source_date='2026-10-03',observed_at=now)
        args += ['--discovery-inventory',ref['path'],'--discovery-inventory-sha256',ref['sha256'],
                 '--discovery-inventory-source-date','2026-10-03']
    scope = SimpleNamespace(value={'source_date':'2026-10-03','persistent_allocation':{'sha256':'a'*64} if persistent else None},admit=lambda *a,**k:None)
    monkeypatch.setattr(FreshnessContract,'load',lambda path:scope)
    from race_collection import live_freshness_contract
    monkeypatch.setattr(live_freshness_contract,'install_request_guard',lambda scope:None)
    monkeypatch.setattr(refresh,'_browser_type',lambda:object)
    class ReachedAcquisition(BaseException):pass
    def browser(*a,**kw):raise ReachedAcquisition
    monkeypatch.setattr(refresh,'_refresh_browser',browser)
    with pytest.raises(ReachedAcquisition if allowed else ValueError):
        refresh.refresh_prejump_upcoming(refresh.build_parser().parse_args(args))
