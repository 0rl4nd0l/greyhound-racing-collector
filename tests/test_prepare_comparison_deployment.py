"""Persistent units must outlive authorized natural drain and stay unarmed."""
from pathlib import Path
import pytest
from scripts.prepare_comparison_deployment import units


def fixture_units():
    return units(source=Path('/release'), python=Path('/runtime/bin/python'),
        schedule_config=Path('/private/schedule.APPROVED.json'),
        result_binding=Path('/private/result-binding.APPROVED.json'), mount=Path('/mnt/evidence'))


def test_supervisor_receives_signal_before_descendant_cleanup_deadline():
    unit = fixture_units()['greyhound-comparison-schedule.service']
    properties = dict(line.split('=', 1) for line in unit.splitlines() if '=' in line)
    assert properties['Type'] == 'exec'  # oneshot ignores RuntimeMaxSec
    assert int(properties['RuntimeMaxSec']) > 300 + 5400 + 1860
    assert int(properties['TimeoutStopSec']) > 1860 + 200
    assert properties['KillMode'] == 'mixed'
    assert properties['Restart'] == 'no'  # durable scheduler decides later work
    assert properties['ConditionPathIsMountPoint'] == '/mnt/evidence'
    assert properties['UMask'] == '0077'
    assert 'WantedBy=' not in unit


def test_timer_persistence_does_not_create_a_second_collector():
    generated = fixture_units()
    assert len(generated) == 6
    for name, unit in generated.items():
        if name.endswith('.timer'):
            assert 'Persistent=true' in unit and 'OnBootSec=2min' in unit
        else:
            assert 'shadow_autopilot' not in unit
    assert '-m scripts.run_comparison_result_queue --binding ' in generated['greyhound-comparison-results.service']


def test_reject_systemd_path_expansion():
    with pytest.raises(ValueError, match='unit_path'):
        units(source=Path('/release/%h'), python=Path('/python'),
            schedule_config=Path('/schedule'), result_binding=Path('/binding'), mount=Path('/mnt/evidence'))
