"""Recovery uses current guidance without changing retired installed evidence."""
from pathlib import Path
import hashlib
import json
import subprocess

import pytest
from race_collection import codex_recovery_watch as watch
from tests.test_codex_recovery_watch import case, trigger, write


def test_installed_legacy_hook_is_not_worker_cwd(case):
    cfg, host, package = case
    installed = Path(cfg['runtime_root']).parent / 'old-installed'
    (installed / '.codex').mkdir(parents=True)
    (installed / '.codex/hooks.json').write_text('{"retired":true}')
    host.collector['WorkingDirectory'] = str(installed)
    trigger(cfg, host)
    watch.run_incident(cfg, host)
    assert host.agents[0]['working'] == cfg['agent_working_directory']
    assert str(installed) in host.agents[0]['prompt']
    assert (installed / '.codex/hooks.json').read_text() == '{"retired":true}'


def test_new_incident_distinguishes_configuration_hash_domains(case):
    cfg, host, package = case
    path = Path(cfg['runtime_root']).parent / 'collector.json'
    path.write_text('{"collector":"configuration"}')
    host.collector['ExecStart'] = f'{{ path=/python ; argv[]=/python collector --config {path} ; }}'
    trigger(cfg, host)
    pointer = json.loads((Path(cfg['state_root']) / 'incident.json').read_bytes())
    incident = json.loads((Path(pointer['directory']) / 'incident.json').read_bytes())
    assert incident['watcher_configuration_sha256'] == watch._configuration_sha(cfg)
    assert incident['collector_configuration']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert 'configuration_sha256' not in incident


@pytest.mark.parametrize('changed', ['guidance', 'hook'])
def test_changed_recovery_context_rejected_before_agent(case, changed):
    cfg, host, package = case
    trigger(cfg, host)
    context = Path(cfg['agent_working_directory'])
    if changed == 'guidance':
        (context/'AGENTS.md').write_text('Different policy')
    else:
        (context/'.codex').mkdir()
        (context/'.codex/hooks.json').write_text('{}')
    result = watch.run_incident(cfg, host)
    assert result['status'] == 'RUNNER_FAILED'
    assert host.agents == []


def test_collector_config_changed_after_capture_suppresses_agent(case):
    cfg, host, package = case
    trigger(cfg, host)
    (Path(cfg['runtime_root']).parent/'collector.json').write_text('{"changed":true}')
    assert watch.run_incident(cfg, host)['status'] == 'SUPPRESSED'
    assert host.agents == []


def test_legacy_incident_digest_is_read_without_rewriting(case):
    cfg, host, package = case
    trigger(cfg, host)
    pointer = json.loads((Path(cfg['state_root'])/'incident.json').read_bytes())
    path = Path(pointer['directory'])/'incident.json'
    value = json.loads(path.read_bytes())
    value['configuration_sha256'] = value.pop('watcher_configuration_sha256')
    write(path, value)
    before = path.read_bytes()
    watch.run_incident(cfg, host)
    assert len(host.agents) == 1 and path.read_bytes() == before


def test_real_exercise_command_reads_installed_legacy_context_without_running_it(case):
    import shlex
    cfg, host, package = case
    installed = Path(host.collector['WorkingDirectory'])
    (installed/'.codex').mkdir()
    hook = installed/'.codex/hooks.json';hook.write_text('{"missing_retired_guard":true}')
    def actual_agent(cfg, directory, prompt, mode, working):
        incident = json.loads((directory/'incident.json').read_bytes())
        command = watch._exercise_command(cfg, incident)
        assert command in prompt and mode == 'exercise'
        result = subprocess.run(shlex.split(command), cwd=working, capture_output=True, text=True)
        (directory/'last-message.txt').write_text('CODEX_RECOVERY_EXERCISE_OK')
        events = [{'type':'thread.started'}, {'type':'turn.completed'},
            {'type':'item.completed', 'item':{'type':'command_execution','command':command,
                'exit_code':result.returncode,'aggregated_output':result.stdout}}]
        (directory/'events.private.jsonl').write_text('\n'.join(map(json.dumps, events)))
        return result.returncode
    host.agent = actual_agent
    assert watch.run_incident(cfg, host, exercise=True)['status'] == 'EXERCISE_COMPLETE'
    assert host.starts == 0 and hook.read_text() == '{"missing_retired_guard":true}'


def test_printf_marker_without_context_command_is_not_an_exercise(case):
    cfg, host, package = case
    original = host.agent
    def fake(cfg, directory, prompt, mode, working):
        rc = original(cfg, directory, prompt, mode, working)
        p = directory/'events.private.jsonl'
        events = [json.loads(line) for line in p.read_text().splitlines()]
        events[1]['item']['command'] = 'printf fake-marker'
        p.write_text('\n'.join(map(json.dumps, events)))
        return rc
    host.agent = fake
    assert watch.run_incident(cfg, host, exercise=True)['status'] == 'AGENT_FAILED'
