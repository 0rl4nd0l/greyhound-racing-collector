from pathlib import Path
import json
import pytest
from scripts.prepare_frozen_comparison_activation import prepare
from src.predictor.future_comparison import load_plan
from src.predictor.on_demand import sha256_file


def test_preparation_creates_no_runtime_and_cannot_activate(tmp_path):
    root=tmp_path/'uncreated-programme';output=tmp_path/'review'
    prepare(starts_at='2099-10-01T12:00:00+10:00',programme_root=root,prediction_output_root=tmp_path/"existing-predictions",output=output,reservation_sha256='a'*64)
    assert not root.exists()
    plan=output/'plan.prepared.json'
    with pytest.raises(ValueError,match='not_activated'):load_plan(plan,sha256_file(plan))
    value=json.loads(plan.read_bytes())
    assert value['activated_at'] is None and value['fixed_calendar_days']==112
