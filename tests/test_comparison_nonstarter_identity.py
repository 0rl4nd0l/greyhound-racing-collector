"""Explicit source nonstarters may not alter the frozen active runner field."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts.autonomous_official_result_capture import comparison_runner_identity_error
from scripts.ingest_results_for_date import SourceResult


def case():
    candidate = SimpleNamespace(participants=[{'box_number': 1, 'dog_name': 'Invented Alpha'},
                                               {'box_number': 2, 'dog_name': 'Invented Beta'}])
    result = SourceResult(source='thedogs_official', status='resulted', source_url='https://example.invalid/race',
        positions_by_box={1: 1, 2: 2}, raw_order=[1, 2],
        dog_names_by_box={1:'Invented Alpha', 2:'Invented Beta', 8:'Invented Nonstarter'},
        terminal_status_by_box={8:'SCR'}, runner_identity_rows_complete=True)
    return candidate, result


def test_nonstarter_filter_does_not_mutate_or_remap_frozen_field():
    candidate, result = case()
    before = deepcopy((candidate, result))
    assert comparison_runner_identity_error(candidate, result) is None
    assert candidate == before[0] and result == before[1]


@pytest.mark.parametrize('mutation', ['unknown_status','missing_status','extra_finish','missing_frozen_name',
    'changed_frozen_name','unnamed_extra','additional_active','unidentified_terminal','reserve','rejected_reserve'])
def test_uncorroborated_field_change_remains_rejected(mutation):
    candidate, result = case()
    if mutation == 'unknown_status':result.terminal_status_by_box[8] = 'DNF'
    elif mutation == 'missing_status':result.terminal_status_by_box = {}
    elif mutation == 'extra_finish':result.positions_by_box[8] = 3
    elif mutation == 'missing_frozen_name':del result.dog_names_by_box[1]
    elif mutation == 'changed_frozen_name':result.dog_names_by_box[1] = 'Different Invented Runner'
    elif mutation == 'unnamed_extra':result.dog_names_by_box[8] = ''
    elif mutation == 'additional_active':result.dog_names_by_box[7] = 'Additional Invented Runner'
    elif mutation == 'unidentified_terminal':result.terminal_status_by_box[7] = 'SCR'
    elif mutation == 'reserve':result.reserve_box_remappings = [{'original_box_number':9,'target_box_number':1}]
    else:result.rejected_reserve_box_remappings = [{'reason':'invented ambiguity'}]
    assert comparison_runner_identity_error(candidate, result) == 'comparison_official_runner_identity_mismatch'

from scripts import ingest_results_for_date as ingest

def row(box, name, status):
    return ('<tr class="race-runner"><td class="race-runners__finish-position">'+status+'</td>'
            '<td class="race-runners__box"><sprite-svg name="rug_'+str(box)+'"></sprite-svg></td>'
            '<td class="race-runners__name"><a>'+name+'</a></td></tr>')


@pytest.mark.parametrize('invalid_extra', ['duplicate_box', 'unnamed_unclassified', 'missing_box', 'missing_position_cell'])
def test_extra_nonstarter_does_not_hide_unproved_rows(invalid_extra):
    participants = [{'box_number':n, 'dog_name':'Invented '+str(n)} for n in (1,2,3)]
    candidate = SimpleNamespace(participants=participants)
    text = '<table class="race-runners--result">'+''.join(row(n, 'Invented '+str(n), pos)
        for n,pos in zip((1,2,3), ('1st','2nd','3rd')))+row(8,'Invented Nonstarter','SCR')
    if invalid_extra == 'duplicate_box': text += row(1,'Invented 1','1st')
    elif invalid_extra == 'unnamed_unclassified': text += row(7,'','')
    elif invalid_extra == 'missing_box': text += row(7,'Invented Unknown','SCR').replace('rug_7','rug_unknown')
    else: text += row(7,'Invented Unknown','SCR').replace('race-runners__finish-position','unrecognized-cell')
    text += '</table>'
    selected = ingest.TheDogsResultFetcher(None)._result_from_html(candidate,'https://invented.invalid/never-requested',text)
    assert comparison_runner_identity_error(candidate, selected) == 'comparison_official_runner_identity_mismatch'


def test_source_row_completeness_pass_propagates_deadline_instead_of_fallback(monkeypatch):
    import bs4
    from src.predictor.comparison_result_runtime import ACTIVE
    class Guard:
        expired = False
        def check_deadline(self):
            if self.expired:
                raise ValueError('RESULT_DEADLINE_EXPIRED')
    guard = Guard()
    native = bs4.BeautifulSoup
    calls = []
    raw_done = []
    native_rows = ingest.parse_thedogs_result_html_runner_rows
    def raw_rows(markup):
        value = native_rows(markup)
        raw_done.append(True)
        return value
    monkeypatch.setattr(ingest, 'parse_thedogs_result_html_runner_rows', raw_rows)
    def parser(*args, **kwargs):
        parsed = native(*args, **kwargs)
        if raw_done:
            calls.append(True)
            guard.expired = True
        return parsed
    monkeypatch.setattr(bs4, 'BeautifulSoup', parser)
    candidate = SimpleNamespace(participants=[{'box_number':1,'dog_name':'Invented One'}])
    markup = '<table class="race-runners--result">'+row(1,'Invented One','1st')+row(8,'Invented Nonstarter','SCR')+'</table>'
    token = ACTIVE.set(guard)
    try:
        with pytest.raises(ValueError, match='^RESULT_DEADLINE_EXPIRED$'):
            ingest.TheDogsResultFetcher(None)._result_from_html(candidate, 'https://example.invalid', markup)
        assert len(calls) == 1
    finally:
        ACTIVE.reset(token)
