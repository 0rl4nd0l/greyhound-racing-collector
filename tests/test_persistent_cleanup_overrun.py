"""Late cleanup remains truthful engineering consumption without poisoning history."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from race_collection.freshness_campaign import Campaign
from race_collection.persistent_authority import stamp
from tests.test_persistent_operation_authority import case, campaign, ledger
from tests.fixtures.persistent_operation_case import put


@pytest.mark.parametrize('whole_allocation',[False,True])
def test_late_close_preserves_full_usage_violation_and_next_day_isolation(case,whole_allocation):
    c=campaign(case)
    start=stamp(case['allocation']['starts_at']) if whole_allocation else case['clock'].now(timezone.utc)
    deadline=stamp(case['allocation']['cleanup_by'])
    c.begin('owned-day',now=start,deadline=deadline)
    closed=deadline+timedelta(seconds=1)
    c.close('owned-day',now=closed)
    prior=ledger(case)
    used=c.persistent_usage(prior,selected=True)
    assert used['live_seconds']==(closed-start).total_seconds()
    assert used['violations'][0]['category']=='PERSISTENT_LEASE_CLEANUP_OVERRUN'
    assert used['violations'][0]['overrun_seconds']==1
    assert Campaign(case['root']).programme_usage(prior)['live_seconds']==0
    c.close('owned-day',now=closed+timedelta(seconds=5))
    assert ledger(case)==prior  # Restart cannot rewrite the original elapsed time.

    a=deepcopy(case['allocation']);a.update(racing_date='2026-10-04',
        allocation_id=case['standing_ref']['sha256']+':2026-10-04',
        issued_at='2026-10-04T08:30:00+11:00',starts_at='2026-10-04T09:00:00+11:00',
        ends_at='2026-10-05T01:00:00+11:00',cleanup_by='2026-10-05T01:30:00+11:00')
    for key in ('state_root','prediction_root'):a[key]=str(Path(case['standing'][key])/'days/2026-10-04')
    ref=put(case['root']/'next-date.json',a)
    case['clock'].current=datetime.fromisoformat('2026-10-04T10:00:00+11:00')
    next_day=Campaign(case['root'],persistent_allocation=ref)
    next_day.request()
    assert next_day.persistent_usage(ledger(case),selected=True)['python']==1
    assert next_day.persistent_usage(ledger(case),selected=True)['violations']==[]
    assert len(next_day.persistent_usage(ledger(case))['violations'])>=1
    with pytest.raises(ValueError,match='window_closed'): c.request()


@pytest.mark.parametrize('change',['undercharged','unclosed_overrun','nonfinite','false_closed_at'])
def test_overrun_reporting_does_not_accept_forged_charge(case,change):
    c=campaign(case);start=case['clock'].now(timezone.utc);deadline=stamp(case['allocation']['cleanup_by'])
    c.begin('owned-day',now=start,deadline=deadline)
    if change!='unclosed_overrun':c.close('owned-day',now=deadline+timedelta(seconds=1))
    value=ledger(case);row=value['launches']['owned-day']
    if change=='undercharged':row['charged_seconds']-=1
    elif change=='unclosed_overrun':row['charged_seconds']+=1
    elif change=='nonfinite':row['charged_seconds']=float('inf')
    else:row['closed_at']=(deadline-timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError,match='persistent_launch_time_invalid'):
        c.persistent_usage(value)
