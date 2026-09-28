from datetime import datetime,timezone
from types import SimpleNamespace
from copy import deepcopy
import pytest
from src.predictor.comparison_results import ComparisonResultSource
from src.operator_ui.journal_results import OfficialResultSource
from scripts.evaluate_frozen_comparison import summarize
from src.predictor.comparison_missingness import paired_bounds


def fixture():
    jump='2026-11-01T03:00:00+00:00';rid='Race 1 - SAN - 2026-11-01';url='https://www.thedogs.com.au/racing/sandown/2026-11-01/1/test'
    job=SimpleNamespace(input=SimpleNamespace(race_id=rid,jump_timestamp=jump,ordered_runners=[{'box':i,'name':f'Invented{i}'} for i in (1,2,3)]))
    bundle=SimpleNamespace(result={'race':{'url':url,'race_date':'2026-11-01','race_number':1,'venue':'SAN'},'generated_at':'2026-11-01T02:55:00+00:00'})
    race={'source':'thedogs_official','status':'resulted','source_url':url,'race_id':rid,'race_date':'2026-11-01','race_number':1,'venue':'SAN',
        'captured_at':'2026-11-01T03:20:00+00:00','start_datetime':jump,'winner_box':1,'winner_name':'Invented1','position_count':3,'participant_count':3,'box_order':[1,2,3]}
    rows=[{**{k:race[k] for k in ('source','source_url','race_id','race_date','race_number','venue','captured_at')},'box_number':i,'dog_name':f'Invented{i}','finish_position':p,'is_winner':p==1} for i,p in [(1,1),(2,1),(3,3)]]
    return job,bundle,[race],rows,datetime(2026,11,1,4,tzinfo=timezone.utc)


def test_dead_heat_preserves_source_and_field_validation_without_production_change():
    args=fixture();original=deepcopy(args[2:4])
    with pytest.raises(ValueError,match='FINISH_AMBIGUOUS'):OfficialResultSource._validate(*args)
    ComparisonResultSource._validate(*args)
    assert args[2:4]==original
    args[3][1]['dog_name']='Wrong Runner'
    with pytest.raises(ValueError,match='IDENTITY_MISMATCH'):ComparisonResultSource._validate(*args)


def test_invalid_competition_ranking_and_late_evidence_reject():
    args=fixture();args[3][2]['finish_position']=2
    with pytest.raises(ValueError,match='FINISH_INVALID'):ComparisonResultSource._validate(*args)
    args=fixture();late=datetime(2026,11,1,3,10,tzinfo=timezone.utc)
    with pytest.raises(ValueError,match='TIMESTAMP_MISMATCH'):ComparisonResultSource._validate(*args[:4],late)


def test_fractional_scoring_and_sparse_blocks_cannot_claim_precision():
    probabilities={m:[.4,.35,.25] for m in ('market','production','residual_box','residual_half')}
    probabilities['residual_box']=[.45,.4,.15]
    rows=[{'date':'2026-11-01','venue':'SAN','boxes':[1,2,3],'outcome':[.5,.5,0],'probabilities':probabilities}]
    summary=summarize(rows,replicates=30)
    assert summary['metrics']['market']['top_choice_accuracy']==.5
    assert not summary['paired']['residual_box-minus-market']['both_upper_bounds_below_zero']
    bounds=paired_bounds(rows,replicates=30)
    assert bounds['resolved']==1 and bounds['unresolved']==0
    assert not bounds['paired']['residual_box-minus-market']['resampling']['week']['inferentially_usable']


def test_existing_result_ingestion_keeps_dead_heats_opt_in(tmp_path):
    import sqlite3
    from scripts.autonomous_official_result_capture import append_official_result_evidence_to_db
    job,bundle,races,rows,now=fixture()
    database=tmp_path/'results.db'
    with sqlite3.connect(database):pass
    artifacts={'race_rows':races,'runner_rows':rows}
    default=append_official_result_evidence_to_db(db_path=database,artifact_rows=artifacts,output_dir=tmp_path,execute=False)
    assert default['blocked_race_rows']==1
    selected=append_official_result_evidence_to_db(db_path=database,artifact_rows=artifacts,output_dir=tmp_path,execute=True,allow_dead_heats=True)
    assert selected['inserted_race_rows']==1
    result=ComparisonResultSource(database).read(job,bundle,now=now)
    assert result['state']=='RESULT_AVAILABLE',result
    assert sum(r['is_winner'] for r in result['evidence']['runner_rows'])==2


def test_official_summary_may_name_either_co_winner():
    args=fixture();args[2][0].update(winner_box=2,winner_name='Invented2')
    ComparisonResultSource._validate(*args)
