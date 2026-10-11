import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from rerun_frozen_live_reconciliation import paired_bounds, verify_originals, verify_frozen_inputs


def forecast():
    return {'race_id':'R1','model_version':'hash:full','runners':[
        {'runner_id':'a','probability':.8,'decimal_odds':2.},
        {'runner_id':'b','probability':.2,'decimal_odds':2.}]}


def test_unseen_outcome_bounds_are_exact_and_not_outcomes():
    result=paired_bounds([forecast()],[])
    assert result['known_outcomes']==0
    assert result['conditional_mean_difference_bounds']['delta_log_loss']==pytest.approx([math.log(.625),math.log(2.5)])
    assert result['conditional_mean_difference_bounds']['delta_brier']==pytest.approx([-.42,.78])
    assert result['conditional_mean_difference_bounds']['delta_top_credit']==[-.5,.5]
    assert paired_bounds([],[])['conditional_mean_difference_bounds']['delta_log_loss'] is None


def test_observed_winner_collapses_bounds():
    f=forecast()
    result={'race_id':'R1','model_version':'hash:full','runners':[{'runner_id':'a','winner':1},{'runner_id':'b','winner':0}]}
    bound=paired_bounds([f],[result])
    assert bound['known_outcomes']==1
    assert bound['conditional_mean_difference_bounds']['delta_log_loss']==pytest.approx([math.log(.625)]*2)


def test_reconstructed_probability_or_earlier_forecast_rejected():
    original={'original':{'race_id':'R1','record_key':'sealed','model_sha256':'hash','score_timestamp':'t1','jump_timestamp':'t2','predictions':[{'runner_id':'a','box_number':1,'full_probability':.8,'strict_win_odds':2.}]}}
    r={'race_id':'R1','prediction_id':'sealed','model_version':'hash:full','predicted_at':'t1','jump_at':'t2','runners':[{'runner_id':'a','box':1,'probability':.8,'decimal_odds':2.}]}
    verify_originals([r],[original])
    r['runners'][0]['probability']=.81
    with pytest.raises(ValueError,match='changed'):verify_originals([r],[original])
    r['prediction_id']='later_replay'
    with pytest.raises(ValueError,match='identity'):verify_originals([r],[original])


def test_repointed_freeze_and_duplicate_population_rejected():
    forecasts=[{'race_id':str(i)} for i in range(210)]
    plan={'dataset_sha256':'data'}
    census={'dataset_sha256':'data','plan_sha256':'plan'}
    verify_frozen_inputs(forecasts,plan,census,'data','plan')
    with pytest.raises(ValueError,match='dataset_hash'):
        verify_frozen_inputs(forecasts,plan,census,'other','plan')
    with pytest.raises(ValueError,match='plan_hash'):
        verify_frozen_inputs(forecasts,plan,census,'data','other')
    with pytest.raises(ValueError,match='population'):
        verify_frozen_inputs(forecasts[:-1]+[forecasts[0]],plan,census,'data','plan')
