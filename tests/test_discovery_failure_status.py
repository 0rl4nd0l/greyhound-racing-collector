"""Exercise real discovery/refresh handling with invented transport responses."""
from datetime import datetime
from types import SimpleNamespace
import pytest
import requests


@pytest.mark.parametrize('response_kind',['connection','connection_response','403','429','empty'])
def test_discovery_error_is_not_success_or_an_automatic_retry(tmp_path,monkeypatch,response_kind):
    from upcoming_race_browser import UpcomingRaceBrowser
    from scripts import refresh_prejump_upcoming as refresh
    browser=UpcomingRaceBrowser.__new__(UpcomingRaceBrowser)
    browser.base_url='https://invented.invalid'
    calls=[]
    def get(url,**kwargs):
        calls.append(url)
        if response_kind=='connection':raise requests.ConnectionError('SYNTHETIC')
        if response_kind=='connection_response':
            response=requests.Response()
            response.status_code=429
            response.headers={'Retry-After':'60'}
            raise requests.ConnectionError('SYNTHETIC',response=response)
        return SimpleNamespace(status_code=200 if response_kind=='empty' else int(response_kind),
                               content=b'<html></html>',close=lambda:None)
    browser.session=SimpleNamespace(get=get)
    browser.get_races_for_date=lambda date:browser._scrape_live_races_for_date(date.isoformat())
    # Deliberately offer a cached race: a failed fresh discovery cannot use it.
    cached=[] if response_kind=='empty' else [{'race_number':1,'date':datetime.now().date().isoformat(),
        'race_time':'23:59','venue':'INVENTED','url':'https://invented.invalid/race'}]
    browser._get_cached_races_for_date=lambda date:cached
    monkeypatch.setattr(refresh,'_refresh_browser',lambda *args,**kwargs:browser)
    browser.download_race_csv=lambda *args,**kwargs:pytest.fail('failed discovery must not download cached races')
    args=refresh.build_parser().parse_args(['--upcoming-dir',str(tmp_path),'--days-ahead','0','--require-safe-metadata'])
    report=refresh.refresh_prejump_upcoming(args)
    assert len(calls)==1
    assert report['selected_count']==report['current_index_race_count']==0
    if response_kind=='empty':
        assert report['status']=='SUCCESS'
        assert report['next_preferred_window']['status']=='NO_RACES_FOUND'
    else:
        assert report['status']=='DISCOVERY_FAILED'
        assert report['next_preferred_window']['status']=='DISCOVERY_FAILED'
        assert report['next_preferred_window']['recommended_rerun_after_local'] is None
        failure=report['discovery_failures'][0]
        assert failure['error_type']==('ConnectionError' if response_kind.startswith('connection') else 'HTTPStatusError')
        if response_kind=='connection':assert 'http_status' not in failure
        elif response_kind=='connection_response':
            assert failure['http_status']==429
            assert failure['source_retry_headers']=={'retry-after':'60'}
        else:assert failure['http_status']==int(response_kind)
