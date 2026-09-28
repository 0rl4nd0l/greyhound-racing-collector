import itertools
import numpy as np
from src.predictor.comparison_missingness import paired_bounds


def race(day,winner=None):
    return {'date':day,'winner':winner,'probabilities':{'market':[.7,.3],'production':[.6,.4],'residual_box':[.65,.35],'residual_half':[.67,.33]}}


def test_unknown_winner_bounds_cover_every_completion_and_void():
    a=[race('2026-11-01',0),race('2026-11-02')]
    bound=paired_bounds(a,replicates=50)['paired']
    for winner in (0,1):
        a[1]['winner']=winner
        complete=paired_bounds(a,replicates=50)['paired']
        for contrast,value in bound.items():
            lohi=np.asarray(value['identified_mean_bounds']); point=np.asarray(complete[contrast]['identified_mean_bounds'])[:,0]
            assert (point>=lohi[:,0]-1e-12).all() and (point<=lohi[:,1]+1e-12).all()
    a[1].update(winner=None,official_void=True)
    complete=paired_bounds(a,replicates=50)['paired']
    for contrast,value in bound.items():
        lohi=np.asarray(value['identified_mean_bounds']);point=np.asarray(complete[contrast]['identified_mean_bounds'])[:,0]
        assert (point>=lohi[:,0]-1e-12).all() and (point<=lohi[:,1]+1e-12).all()


def test_date_and_week_dependence_units_and_no_fake_precision():
    data=[race('2026-11-02',0),race('2026-11-03',1),race('2026-11-09')]
    result=paired_bounds(data,replicates=100)
    assert result['unresolved']==1
    contrast=result['paired']['residual_box-minus-market']
    assert contrast['resampling']['date']['blocks']==3
    assert contrast['resampling']['week']['blocks']==2
    assert not contrast['resampling']['date']['both_upper_bounds_below_zero']
