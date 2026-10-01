import io
from types import SimpleNamespace
import pytest
from race_collection.operational_prediction import Supervisor


@pytest.mark.parametrize('method', ['tick', 'drain'])
def test_verified_grade_rejection_does_not_abort_other_races(tmp_path, method):
    supervisor = Supervisor(tmp_path, {}, None)
    supervisor.child = SimpleNamespace(poll=lambda: 3, returncode=3, wait=lambda timeout: 3)
    supervisor.log = io.StringIO()
    getattr(supervisor, method)()
    assert supervisor.child is None
    assert supervisor.log.closed


@pytest.mark.parametrize('method', ['tick', 'drain'])
def test_unknown_failure_still_stops_and_preserves_consumption(tmp_path, method):
    supervisor = Supervisor(tmp_path, {}, None)
    supervisor.child = SimpleNamespace(poll=lambda: 2, returncode=2, wait=lambda timeout: 2)
    supervisor.log = io.StringIO()
    with pytest.raises(ValueError, match='operational_prediction_failed_preserved_consumption'):
        getattr(supervisor, method)()
