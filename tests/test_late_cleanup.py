"""Exercise the actual packager and runtime contract across midnight cleanup."""
from datetime import datetime, timedelta
from copy import deepcopy
from pathlib import Path
import pytest
from tests.test_late_evening_incident_authority import late, single
from tests.fixtures.incident_engineering_case import put
from tests import test_incident_service_startup as startup
from race_collection.live_freshness_contract import FreshnessContract


def test_late_package_and_contract_allow_only_bound_cleanup(tmp_path, monkeypatch, late):
    original_case = startup.case
    def late_case(root):
        value, ref, comparison, comparison_ref = original_case(root)
        template, _ = late
        for key in ('schema_version','authority_reference','issued_at','late_window_amendment',
                    'live_first_amendment','slots','collection_stop_at','cleanup_deadline',
                    'result_deadline','limits_basis','max_capture_attempts_per_window',
                    'max_prediction_logical_requests_per_window','max_result_requests_per_window',
                    'max_result_operations_per_window','max_python_requests_per_window',
                    'max_browser_navigations_per_window'):
            value[key] = deepcopy(template[key])
        slot = value['slots'][0]
        slot['cleanup_by'] = (datetime.fromisoformat(slot['ends_at']) + timedelta(seconds=1860)).isoformat()
        ref = put(Path(ref['path']), value)
        comparison.update(incident_authority=ref, authority_reference=value['authority_reference'],
            activated_at=value['issued_at'], starts_at=slot['starts_at'], ends_at=slot['cleanup_by'])
        return value, ref, comparison, put(Path(comparison_ref['path']), comparison)
    monkeypatch.setattr(startup, 'case', late_case)
    monkeypatch.setattr(startup, 'STAMP', datetime.fromisoformat('2026-10-02T22:01:00+10:00'))
    original_contract = startup.FreshnessContract
    def current_date_contract(value):
        value['source_date'] = '2026-10-02'
        return original_contract(value)
    monkeypatch.setattr(startup, 'FreshnessContract', current_date_contract)
    package, plan, contract, ref, gate, http = startup.incident_service.__wrapped__(tmp_path, monkeypatch)
    scope = FreshnessContract(contract)
    assert scope.end.date().isoformat() == '2026-10-02'
    assert (scope.end + timedelta(seconds=contract['cleanup_seconds'])).date().isoformat() == '2026-10-03'
    with pytest.raises(ValueError, match='operating_scope_closed'):
        scope.admit(datetime.fromisoformat('2026-10-03T00:00:00+10:00'), seconds=1)
    unbound = deepcopy(contract)
    del unbound['incident_authority']
    with pytest.raises(ValueError, match='one_date_scope_required'):
        FreshnessContract(unbound)
    changed = deepcopy(contract)
    changed['incident_authority']['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        FreshnessContract(changed)
