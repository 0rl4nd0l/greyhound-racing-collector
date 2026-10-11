"""Contract tests: identity safety and chronology of the retrospective diagnostic."""
import numpy as np
import pytest
from scripts.run_retrospective_sp_benchmark import parse_sp, fit_correction, apply, metrics


def table(price='$2.00', dog='123', box='1', finish='1st'):
    return (f'<table class="race-runners--result"><tr class="race-runner">'
            f'<td class="race-runners__finish-position">{finish}</td>'
            f'<td class="race-runners__box"><sprite-svg name="rug_{box}"></sprite-svg></td>'
            f'<blackbook-dog data-dog-id="{dog}"></blackbook-dog>'
            f'<td class="race-runners__starting-price">{price}</td></tr></table>').encode()


def test_sp_requires_exact_identity_and_actual_numeric_price():
    assert parse_sp(table(),[(1,'123')]) == [2.0]
    for raw,expected in [(table(),[(1,'124')]),(table('SP'),[(1,'123')]),(table('$0'),[(1,'123')]),(table(),[(2,'123')])]:
        with pytest.raises(ValueError):parse_sp(raw,expected)


def test_duplicate_result_box_rejected():
    raw=table().replace(b'</table>',table().split(b'>',1)[1])
    with pytest.raises(ValueError,match='DUPLICATE_BOX'):parse_sp(raw,[(1,'123')])


def race(date,winner):
    return {'source_race_key':date,'race_date':date,'track':'SYNTHETIC','runners':[
        {'y':int(i==winner),'probabilities':{'reported_sp':p,'form':q}}
        for i,(p,q) in enumerate(zip([.7,.3],[.3,.7]))]}


def test_fit_correction_is_bounded_and_future_labels_cannot_affect_it():
    past=[race('2025-01-01',0),race('2025-01-02',1)]
    fit=fit_correction(past,'form')
    future=[race('2025-01-03',0)]
    apply(future,'combo',fit)
    p=[r['probabilities']['combo'] for r in future[0]['runners']]
    future[0]['runners'][0]['y']=0;future[0]['runners'][1]['y']=1
    apply(future,'combo',fit)
    assert [r['probabilities']['combo'] for r in future[0]['runners']]==p
    assert .5<=fit['coefficients'][0]<=1.5
    assert 0<=fit['coefficients'][1]<=.5
    assert np.isclose(sum(p),1)
    assert max(fit['train_dates'])<'2025-01-03'


def test_equal_race_brier_and_tie_credit():
    r=race('2025-01-01',0)
    for v in r['runners']:v['probabilities']['uniform']=.5
    metric,_=metrics([r],'uniform')
    assert metric['log_loss']==pytest.approx(np.log(2))
    assert metric['brier']==.5
    assert metric['top1']==.5


def test_extra_active_result_runner_rejected_but_explicit_nonstarter_allowed():
    def joined(second):
        return table().replace(b'</table>',second.split(b'>',1)[1])
    with pytest.raises(ValueError,match='ACTIVE_FIELD_MISMATCH'):
        parse_sp(joined(table('$3','456','2','2nd')),[(1,'123')])
    assert parse_sp(joined(table('$3','456','2','SCR')),[(1,'123')])==[2.]
    with pytest.raises(ValueError,match='ACTIVE_STATUS_UNQUALIFIED'):
        parse_sp(joined(table('$3','456','2','')),[(1,'123')])


@pytest.mark.parametrize('probabilities,winners',[
    ([.7,.7],[1,0]),([float('nan'),.3],[1,0]),([0,1],[1,0]),
    ([.7,.3],[1,1]),([.7,.3],[0,0]),([.4,.3,.3],[1,1,-1]),
])
def test_metrics_reject_malformed_race(probabilities,winners):
    r=race('2025-01-01',0)
    r['runners']=[{'y':y,'probabilities':{'bad':p}} for p,y in zip(probabilities,winners)]
    with pytest.raises(ValueError):metrics([r],'bad')


def test_dynamic_checks_base_manifest_before_decoding(tmp_path,monkeypatch):
    import json
    from scripts import run_retrospective_sp_benchmark as benchmark
    base=tmp_path/'base';base.mkdir()
    (base/'predictions.jsonl').write_text('not JSON, must not decode')
    (base/'artifacts.sha256.json').write_text(json.dumps({'predictions.jsonl':'0'*64,'protocol.json':'0'*64}))
    monkeypatch.setattr(benchmark,'lines',lambda path:pytest.fail('decoded before manifest verification'))
    with pytest.raises(ValueError,match='ARTIFACT_HASH_MISMATCH'):
        benchmark.add_dynamic(tmp_path/'out',base,tmp_path/'dynamic')


def test_dynamic_rejects_manifest_omitting_predictions(tmp_path):
    from scripts import run_retrospective_sp_benchmark as benchmark
    base=tmp_path/'base';base.mkdir()
    (base/'artifacts.sha256.json').write_text('{}')
    with pytest.raises(ValueError,match='REQUIRED_ARTIFACT_MISSING'):
        benchmark.add_dynamic(tmp_path/'out',base,tmp_path/'dynamic')
