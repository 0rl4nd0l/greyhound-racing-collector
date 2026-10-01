"""Invented official spelling variants use the same native identity rule as live."""
from pathlib import Path
import hashlib
import json
import sqlite3

import pytest
from tests.test_reconcile_comparison_result_identity import case, run, put, ref, assert_quarantined


def replace_body(case, text):
    body = Path(case['authority']['body']['path'])
    body.write_text(text)
    case['authority']['body'] = ref(body)
    response_path = Path(case['authority']['response']['path'])
    response = json.loads(response_path.read_bytes())
    response.update(sha256=ref(body)['sha256'], bytes=body.stat().st_size)
    case['authority']['response'] = put(response_path, response)


@pytest.mark.parametrize('spelling', ['case', 'spacing', 'apostrophe'])
@pytest.mark.parametrize('nonstarters', [False, True])
def test_native_case_normalization_closes_once_preserving_frozen_names(case, spelling, nonstarters):
    body = Path(case['authority']['body']['path'])
    name = case['job'].input.ordered_runners[0]['name']
    variants = {'case':name.swapcase(), 'spacing':' '.join(name.replace(' ','')),
                'apostrophe':name[0]+"'"+name[1:]}
    text = body.read_text().replace('<a>'+name+'</a>', '<a>'+variants[spelling]+'</a>')
    if nonstarters:
        from tests.test_comparison_nonstarter_identity import row
        text = text.replace('</table>', row(8,'Invented Extra Eight','SCR')
                            +row(10,'Invented Extra Ten','L/SCR')+'</table>')
    replace_body(case, text)
    originals = {Path(case['authority'][key]['path']):case['authority'][key]['sha256']
                 for key in ['body','request','response','failed_report']}
    ledger = Path(case['runtime']['campaign_root'])/'ledger.json'
    ledger_before = ledger.read_bytes()
    result = run(case)
    assert result['status'] == 'CLOSED_FROM_RETAINED_IDENTITY_RECONCILIATION'
    assert result['provider_requests'] == 0 and result['outcomes_released'] is False
    assert ledger.read_bytes() == ledger_before
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == sha for path,sha in originals.items())
    with sqlite3.connect(case['root']/'queue.sqlite3') as db:
        assert db.execute('SELECT state,attempts FROM jobs').fetchone() == ('CLOSED',1)
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM events').fetchone()[0] == 2
    with sqlite3.connect(case['root']/'official-results.sqlite3') as db:
        assert dict(db.execute('SELECT box_number,dog_name FROM autonomous_official_result_evidence_runners')) == {
            runner['box']:runner['name'] for runner in case['job'].input.ordered_runners}
    with pytest.raises(ValueError):
        run(case)
    with sqlite3.connect(case['root']/'queue.sqlite3') as db:
        assert db.execute('SELECT state,attempts FROM jobs').fetchone() == ('CLOSED',1)
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM events').fetchone()[0] == 2


@pytest.mark.parametrize('mutation', ['changed_name', 'empty_name', 'swapped_boxes', 'duplicate', 'reserve'])
def test_native_normalization_never_substitutes_a_frozen_runner(case, mutation):
    from tests.test_comparison_nonstarter_identity import row
    body = Path(case['authority']['body']['path'])
    text = body.read_text()
    first, second = case['job'].input.ordered_runners[:2]
    name = first['name']
    if mutation in {'changed_name','empty_name'}:
        replacement = name.swapcase()+'X' if mutation == 'changed_name' else ''
        text = text.replace('<a>'+name+'</a>', '<a>'+replacement+'</a>')
    elif mutation == 'swapped_boxes':
        text = text.replace('rug_'+str(first['box']), 'RUG_PLACEHOLDER')
        text = text.replace('rug_'+str(second['box']), 'rug_'+str(first['box']))
        text = text.replace('RUG_PLACEHOLDER', 'rug_'+str(second['box']))
    elif mutation == 'duplicate':
        text = text.replace('</table>', text[text.index('<tr'):text.index('</tr>')+5]+'</table>')
    else:
        # The native parser can establish a promoted reserve mapping, but that
        # remains forbidden for this exact frozen comparison field.
        text = text.replace('rug_'+str(first['box']), 'rug_9', 1)
        text = text.replace('<a>'+name+'</a>', '<a>'+name+' (from box '+str(first['box'])+')</a>')
        text = text.replace('</table>', row(first['box'],name,'SCR')+'</table>')
    replace_body(case, text)
    with pytest.raises(ValueError):
        run(case)
    assert_quarantined(case)
    with sqlite3.connect(case['root']/'official-results.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM autonomous_official_result_evidence_races').fetchone()[0] == 0
