"""Durable one-look IO for a separately admitted prospective development cohort.

This module grants no result authority and cannot contact providers. Before use,
the amended retention coordinator must supply a root-verified, selected-only
manifest and exact single-race win-target projections. Old pilot result files
are not silently treated as these projections. Every target and identity proof
must be bound to its original retained closure evidence and exact frozen roster.
The coordinator authenticates allocation/official identities; this layer checks
those bindings and the durable pre-jump forecast chain before opening targets.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

from race_collection import prospective_speed_inputs as inputs
from race_collection import prospective_speed_plan as plan
from race_collection import prospective_speed_runtime as runtime
from race_collection.retained_card_timing_coverage import Reader, instant


class EvaluationIORejected(ValueError):
    """A safe finite category; never echo a private source value."""


def _require(condition, reason):
    if not condition:
        raise EvaluationIORejected(reason)


def _digest(value):
    return hashlib.sha256(runtime.encoded(value)).hexdigest()


class _Checked:
    """Finite cached IO: at most 384 authenticated files / 512 MiB per look."""
    def __init__(self):
        self.cache, self.reads, self.bytes = {}, 0, 0

    def raw(self, ref):
        _require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'REFERENCE_INVALID')
        key = (ref['path'], ref['sha256'])
        if key not in self.cache:
            raw = Reader().read(ref)
            self.reads += 1
            self.bytes += len(raw)
            _require(self.reads <= 384 and self.bytes <= 512 * 1024**2, 'EVALUATION_IO_LIMIT')
            self.cache[key] = raw
        return self.cache[key]

    def json(self, ref):
        return json.loads(self.raw(ref))

    def output_json(self, ref):
        # Forecast benchmark evidence has its separately measured 32 MiB cap.
        # Do not cache these large decoded graphs across the entire cohort.
        value = runtime.read_output(ref)
        self.reads += 1
        self.bytes += Path(ref['path']).stat().st_size
        _require(self.reads <= 384 and self.bytes <= 512 * 1024**2, 'EVALUATION_IO_LIMIT')
        return value


def _at(ref, path):
    _require(Path(ref['path']) == path, 'FORECAST_ROLE_PATH_CHANGED')


def _forecast(reader, entry, job, experiment_plan, population):
    """Prove completion → seal → terminal → payload and its original job claim."""
    race_id = entry['race_id']
    attempt = Path(job['forecast_root']) / hashlib.sha256(race_id.encode()).hexdigest()
    completion_ref = entry['forecast_completion']
    _at(completion_ref, attempt / 'completion.json')
    completion = reader.json(completion_ref)
    _require(completion['status'] == 'SEALED_PREJUMP', 'FORECAST_NOT_SEALED_PREJUMP')
    _at(completion['seal'], attempt / 'seal.json')
    seal = reader.json(completion['seal'])
    _at(seal['terminal'], attempt / 'terminal.json')
    terminal = reader.json(seal['terminal'])
    _require(terminal['race_id'] == race_id and terminal['status'] == 'FORECAST_PAYLOAD_DURABLE'
        and terminal['payload'] == seal['payload'], 'FORECAST_TERMINAL_CHANGED')
    _at(terminal['payload'], attempt / 'forecast.json')
    _at(terminal['claim'], attempt / 'claim.json')
    _at(terminal['execution_job'], attempt / 'job.json')
    claim = reader.json(terminal['claim'])
    execution = reader.json(terminal['execution_job'])
    source_job = reader.json(claim['source_job'])
    _require(claim['mode'] == 'PROSPECTIVE' and claim['race_id'] == race_id
        and execution['execution_mode'] == 'PROSPECTIVE'
        and {k: v for k, v in execution.items() if k not in {'execution_mode', 'forecast_at'}}
            == {k: v for k, v in source_job.items() if k not in {'execution_mode', 'forecast_at'}},
        'FORECAST_JOB_CLAIM_CHANGED')
    _require(execution['plan'] == job['plan']
        and execution['population'] == job['population_by_date'][entry['race_date']],
        'FORECAST_ALLOCATION_CHANGED')
    activation = reader.json(execution['activation'])
    _require(activation.get('status') == 'AUTHORIZED_PROSPECTIVE_DEVELOPMENT'
        and activation.get('plan_sha256') == job['plan']['sha256']
        and activation.get('population_sha256') == execution['population']['sha256']
        and activation.get('candidate_commit') == inputs.FROZEN_CANDIDATE_COMMIT
        and activation.get('additional_source_requests') == 0
        and activation.get('additional_result_requests') == 0
        and activation.get('development_precedence_verified') is True,
        'FORECAST_ACTIVATION_CHANGED')
    payload = reader.output_json(terminal['payload'])
    _require(payload['schema_version'] == 'prospective_frozen_speed_forecast_v1'
        and payload['race_id'] == race_id and payload['race_date'] == entry['race_date']
        and payload['source_member'] == execution['member']
        and payload['original_publication'] == execution['original']
        and instant(payload['forecast_at']) == instant(execution['forecast_at'])
        and payload['baseline_artifact'] == execution['model']
        and payload['baseline_artifact']['sha256'] == inputs.BASELINE_ARTIFACT_SHA256
        and payload['contract'] == inputs.frozen_contract(), 'FORECAST_CANDIDATE_CHANGED')
    times = [claim['claimed_at'], terminal['completed_at'], seal['completed_at'], completion['completed_at']]
    _require(all(instant(a) <= instant(b) for a, b in zip(times, times[1:]))
        and instant(completion['completed_at']) < instant(payload['jump_at']), 'FORECAST_DURABILITY_TIME_CHANGED')
    plan.forecast_admission(experiment_plan, population, race_id,
        cutoff=payload['information_cutoff'], sealed_at=completion['completed_at'],
        input_available_at=payload['original_publication']['original_published_complete_at'])
    predictions = payload['predictions']
    runner_ids = [r['runner_id'] for r in predictions]
    speed = payload['speed_features']['runners']
    _require(runner_ids == [r['runner_id'] for r in speed]
        == [r['runner_id'] for r in payload['speed_packet']['roster']]
        == entry['runner_ids'], 'FORECAST_ROSTER_CHANGED')
    return {'race_id': race_id, 'race_date': entry['race_date'],
        'forecast_status': 'SEALED_PREJUMP', 'label_status': entry['label_status'],
        'cutoff': payload['information_cutoff'], 'sealed_at': completion['completed_at'],
        'input_available_at': payload['original_publication']['original_published_complete_at'],
        'runner_ids': runner_ids,
        'speed_estimates': [r['speed_estimate'] for r in speed],
        'speed_supported': [r['status'] == 'SUPPORTED' for r in speed],
        'probabilities': {name: [r[field] for r in predictions] for name, field in (
            ('market', 'market'), ('baseline', 'baseline'), ('baseline_speed', 'baseline_plus_speed'))},
        'outcome': None, 'result_identity_verified': False}, {'jump_at': payload['jump_at']}


def _failed_forecast(reader, entry, job):
    """Bind an actual failure when one exists; unattempted rows remain explicit."""
    _require(entry['target'] is None
        and entry['identity_proof'] is None and entry['closure_evidence'] is None
        and entry['label_status'] == 'UNREAD_FORECAST_FAILURE', 'FAILED_FORECAST_TARGET_FORBIDDEN')
    attempt = Path(job['forecast_root']) / hashlib.sha256(entry['race_id'].encode()).hexdigest()
    status = entry['forecast_status']
    terminal_ref = entry['forecast_terminal']
    if status == 'INTERRUPTED_PARTIAL_CLAIM':
        _require(terminal_ref is None and entry['forecast_completion'] is None
            and entry['failure_evidence'] is not None, 'PARTIAL_CLAIM_EVIDENCE_MISSING')
        _at(entry['failure_evidence'], attempt / 'interrupted-partial-claim.json')
        partial = reader.json(entry['failure_evidence'])
        _require(set(partial) == {'status', 'claim', 'at'} and partial['status'] == status
            and instant(partial['at']) <= runtime.utc_now()
            and not (attempt/'terminal.json').exists() and not (attempt/'completion.json').exists(),
            'PARTIAL_CLAIM_EVIDENCE_CHANGED')
        _at(partial['claim'], attempt / 'claim.json')
        reader.raw(partial['claim'])  # Authenticate original truncated bytes; never parse or repair them.
        if entry.get('source_job') is not None:
            source_job = reader.json(entry['source_job'])
            _require(source_job['member']['race_id'] == entry['race_id']
                and source_job['plan'] == job['plan']
                and source_job['population'] == job['population_by_date'][entry['race_date']],
                'PARTIAL_CLAIM_SOURCE_ASSIGNMENT_CHANGED')
        # Without a separately bound source job, only selected membership, the
        # exact attempt path and raw claim hash are established. No parsed claim
        # or completed forecast identity is inferred from damaged evidence.
    elif status == 'INTERRUPTED_BEFORE_CLAIM':
        _require(terminal_ref is None and entry['forecast_completion'] is None
            and entry['failure_evidence'] is not None, 'ORPHAN_ATTEMPT_EVIDENCE_MISSING')
        _at(entry['failure_evidence'], attempt / 'interrupted-before-claim.json')
        orphan = reader.json(entry['failure_evidence'])
        source_job = reader.json(orphan['source_job'])
        _require(orphan['status'] == status and source_job['member']['race_id'] == entry['race_id']
            and source_job['plan'] == job['plan']
            and source_job['population'] == job['population_by_date'][entry['race_date']]
            and not any((attempt / name).exists() for name in ('claim.json', 'terminal.json', 'completion.json')),
            'ORPHAN_ATTEMPT_EVIDENCE_CHANGED')
    elif terminal_ref is not None:
        _at(terminal_ref, attempt / 'terminal.json')
        terminal = reader.json(terminal_ref)
        if terminal.get('schema_version') == 'prospective_speed_stage_failure_v1':
            _require(set(terminal) == {'schema_version', 'race_id', 'status', 'reason', 'at',
                'plan_sha256', 'population_sha256'}
                and entry['forecast_completion'] is None
                and entry['failure_evidence'] in (None, terminal_ref)
                and terminal['race_id'] == entry['race_id'] and terminal['status'] == status
                and status in {'UPSTREAM_FORECAST_UNAVAILABLE', 'SPEED_PROCESSING_FAILED', 'UNATTEMPTED_AT_HORIZON'}
                and isinstance(terminal['reason'], str) and bool(terminal['reason'])
                and terminal['plan_sha256'] == _digest(reader.json(job['plan']))
                and terminal['population_sha256'] == _digest(reader.json(job['population_by_date'][entry['race_date']]))
                and instant(terminal['at']) <= runtime.utc_now()
                and not (attempt/'claim.json').exists() and not (attempt/'completion.json').exists(),
                'PRE_FORECAST_FAILURE_BINDING_CHANGED')
            return {key: entry[key] for key in ('race_id', 'race_date', 'forecast_status', 'label_status')} | {
                'outcome': None, 'probabilities': None}
        _require(entry['failure_evidence'] is None, 'UNEXPECTED_FAILURE_EVIDENCE')
        _at(terminal['claim'], attempt / 'claim.json')
        claim = reader.json(terminal['claim'])
        source_job = reader.json(claim['source_job'])
        _require(terminal['race_id'] == claim['race_id'] == entry['race_id']
            and claim['mode'] == 'PROSPECTIVE' and source_job['member']['race_id'] == entry['race_id']
            and source_job['plan'] == job['plan']
            and source_job['population'] == job['population_by_date'][entry['race_date']],
            'FAILED_FORECAST_CLAIM_CHANGED')
        if status in {'LATE_SPEED_SEAL', 'INTERRUPTED_SEAL'}:
            _require(terminal['status'] == 'FORECAST_PAYLOAD_DURABLE', 'FAILED_SEAL_TERMINAL_CHANGED')
            _at(terminal['payload'], attempt / 'forecast.json')
            payload = reader.output_json(terminal['payload'])
            _require(payload['race_id'] == entry['race_id'], 'FAILED_SEAL_PAYLOAD_CHANGED')
            if status == 'LATE_SPEED_SEAL':
                _require(entry['forecast_completion'] is not None, 'LATE_COMPLETION_MISSING')
                _at(entry['forecast_completion'], attempt / 'completion.json')
                completion = reader.json(entry['forecast_completion'])
                _at(completion['seal'], attempt / 'seal.json')
                seal = reader.json(completion['seal'])
                _require(completion['status'] == status and seal['terminal'] == terminal_ref
                    and seal['payload'] == terminal['payload']
                    and instant(completion['completed_at']) >= instant(payload['jump_at']),
                    'LATE_COMPLETION_CHANGED')
            else:
                _require(entry['forecast_completion'] is None and not (attempt/'completion.json').exists(),
                         'INTERRUPTED_SEAL_COMPLETION_EXISTS')
        else:
            _require(entry['forecast_completion'] is None and terminal['status'] == status
                and status not in {'FORECAST_PAYLOAD_DURABLE', 'REPLAY_NOT_LIVE'},
                'FAILED_FORECAST_TERMINAL_CHANGED')
    else:
        _require(status in {'UPSTREAM_FORECAST_UNAVAILABLE', 'UNATTEMPTED_AT_HORIZON'}
            and entry['failure_evidence'] is None and entry['forecast_completion'] is None
            and not attempt.exists(),
                 'FAILED_FORECAST_EVIDENCE_MISSING')
    return {key: entry[key] for key in ('race_id', 'race_date', 'forecast_status', 'label_status')} | {
        'outcome': None, 'probabilities': None}


def _membership(reader, job, experiment_plan):
    populations = [reader.json(ref) for ref in job['populations']]
    date_accounting = reader.json(job['date_accounting'])
    members = {}
    by_date = {}
    for ref, population in zip(job['populations'], populations):
        day = population['local_date']
        _require(day not in by_date, 'POPULATION_DATE_DUPLICATE')
        by_date[day] = ref
        for race_id in population['selected_race_ids']:
            _require(race_id not in members, 'POPULATION_MEMBER_DUPLICATE')
            members[race_id] = population
    # Outcome-free validation uses the same complete date/member accounting gate.
    metadata = [{'race_id': race_id, 'race_date': pop['local_date'],
        'forecast_status': 'METADATA_PREFLIGHT_ONLY', 'label_status': 'UNREAD',
        'outcome': None, 'probabilities': None} for race_id, pop in members.items()]
    plan.evaluate_closed_population(experiment_plan, populations, metadata,
        date_accounting=date_accounting, now=runtime.utc_now().isoformat(),
        collection_terminal=True, closure_terminal=True)
    return populations, date_accounting, members, by_date


def _manifest(reader, job, experiment_plan, members):
    allocation = reader.json(job['allocation'])
    _require(job['allocation'] == experiment_plan['authority']['allocation']
        and allocation['allocation_id'] == experiment_plan['allocation_id']
        and allocation['status'] == 'AUTHORIZED', 'EVALUATION_ALLOCATION_CHANGED')
    manifest = reader.json(job['result_manifest'])
    authority = reader.json(job['evaluation_authority'])
    member_rows = sorted([{'race_id': key, 'race_date': value['local_date']}
                           for key, value in members.items()], key=lambda x: x['race_id'])
    _require(authority.get('schema_version') == 'prospective_speed_evaluation_authority_v1'
        and authority.get('status') == 'AUTHORIZED_SELECTED_DEVELOPMENT_EVALUATION'
        and isinstance(authority.get('authority_reference'), str) and bool(authority['authority_reference'])
        and authority.get('plan') == job['plan'] and authority.get('allocation') == job['allocation']
        and authority.get('result_manifest') == job['result_manifest']
        and authority.get('members_sha256') == _digest(member_rows)
        and authority.get('additional_result_requests') == 0,
        'EVALUATION_RESULT_AUTHORITY_NOT_ADMITTED')
    _require(manifest.get('schema_version') == 'prospective_speed_admitted_result_manifest_v1'
        and manifest.get('status') == 'ROOT_VERIFIED_SELECTED_DEVELOPMENT_CLOSURE'
        and manifest.get('plan') == job['plan'] and manifest.get('allocation') == job['allocation']
        and manifest.get('members_sha256') == _digest(member_rows)
        and manifest.get('collection_terminal') is True and manifest.get('closure_terminal') is True,
        'RESULT_MANIFEST_NOT_ADMITTED')
    entries = manifest['entries']
    _require(isinstance(entries, list) and len(entries) == len(members)
        and {r['race_id'] for r in entries} == set(members), 'RESULT_MANIFEST_DENOMINATOR_CHANGED')
    fields = {'race_id', 'race_date', 'forecast_status', 'forecast_completion', 'forecast_terminal',
        'runner_ids', 'label_status', 'target', 'target_role', 'identity_proof', 'closure_evidence', 'failure_evidence'}
    for entry in entries:
        _require(set(entry) in (fields, fields | {'source_job'})
            and ('source_job' not in entry or entry['forecast_status'] == 'INTERRUPTED_PARTIAL_CLAIM')
            and entry['race_date'] == members[entry['race_id']]['local_date'],
                 'RESULT_MANIFEST_RECORD_INVALID')
        if entry['label_status'] in plan.VERIFIED_LABELS:
            _require(entry['forecast_status'] == 'SEALED_PREJUMP'
                and entry['target_role'] == 'SELECTED_DEVELOPMENT_WIN_TARGET'
                and entry['failure_evidence'] is None
                and all(entry[k] is not None for k in ('target', 'identity_proof', 'closure_evidence')),
                'RESULT_TARGET_ROLE_NOT_ADMITTED')
        else:
            _require(all(entry[k] is None for k in ('target', 'target_role', 'identity_proof', 'closure_evidence')),
                     'UNVERIFIED_TARGET_REFERENCE_FORBIDDEN')
    return entries


def _target(reader, entry, row, payload, job, experiment_plan):
    """Only exact selected single-race projections; no database or directory search."""
    proof = reader.json(entry['identity_proof'])
    _require(proof.get('schema_version') == 'prospective_speed_result_identity_proof_v1'
        and proof.get('status') == 'VERIFIED_SELECTED_DEVELOPMENT_WIN_TARGET'
        and proof.get('race_id') == row['race_id'] and proof.get('runner_ids') == row['runner_ids']
        and proof.get('target') == entry['target'] and proof.get('closure_evidence') == entry['closure_evidence']
        and proof.get('forecast_completion') == entry['forecast_completion']
        and proof.get('allocation') == job['allocation'] and proof.get('plan') == job['plan'],
        'OFFICIAL_IDENTITY_PROOF_CHANGED')
    reader.raw(entry['closure_evidence'])  # Verify retained bytes; never perform another fetch.
    target = reader.json(entry['target'])
    fields = {'schema_version', 'role', 'race_id', 'race_date', 'jump_at', 'runner_ids', 'outcome',
        'label_status', 'official_observed_at', 'closure_evidence', 'plan', 'allocation'}
    _require(set(target) == fields and target['schema_version'] == 'prospective_speed_verified_win_target_v1'
        and target['role'] == 'SELECTED_DEVELOPMENT_WIN_TARGET'
        and target['race_id'] == row['race_id'] and target['race_date'] == row['race_date']
        and target['jump_at'] == payload['jump_at'] and target['runner_ids'] == row['runner_ids']
        and target['label_status'] == row['label_status']
        and target['closure_evidence'] == entry['closure_evidence']
        and target['plan'] == job['plan'] and target['allocation'] == job['allocation'],
        'OFFICIAL_TARGET_IDENTITY_CHANGED')
    _require(instant(payload['jump_at']) <= instant(target['official_observed_at'])
        <= instant(experiment_plan['result_requests_stop_at']) <= runtime.utc_now(),
        'OFFICIAL_TARGET_TIMING_INVALID')
    return {**row, 'outcome': target['outcome'], 'result_identity_verified': True}


def run_evaluation(job_reference, output_root):
    """Actual-clock one-look execution. No public time or authority bypass exists.

Any claim, including an interrupted claim before the first result read, consumes
the look. Repeat invocations return recorded state without re-reading targets.
Only root can issue the admitted result manifest under an approved allocation.
"""
    reader = _Checked()
    job = reader.json(job_reference)
    _require(job.get('schema_version') == 'prospective_speed_evaluation_job_v1', 'JOB_INVALID')
    experiment_plan = reader.json(job['plan'])
    now = runtime.utc_now()
    gate = plan.evaluation_gate(experiment_plan, now=now.isoformat(),
        collection_terminal=True, closure_terminal=True)
    if gate != 'EVALUATION_DUE':
        return {'status': gate, 'result_accesses_consumed': 0}
    root = Path(output_root)
    with runtime.exclusive(root):
        info = root.stat()
        _require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) & 0o077 == 0,
                 'EVALUATION_OUTPUT_NOT_PRIVATE')
        claim_path, terminal_path = root / 'evaluation-claim.json', root / 'evaluation-terminal.json'
        if claim_path.exists():
            claim_reference = runtime.reference(claim_path)
            try:
                claim = reader.json(claim_reference)
            except json.JSONDecodeError:
                if not terminal_path.exists():
                    runtime.put_new(terminal_path, {'status': 'PARTIAL_EVALUATION_CLAIM_NO_RETRY',
                        'claim': claim_reference, 'at': runtime.utc_now().isoformat()})
                return {'status': 'PARTIAL_EVALUATION_CLAIM_NO_RETRY',
                    'terminal': runtime.reference(terminal_path), 'result_accesses_consumed': 0}
            _require(claim['job'] == job_reference, 'EVALUATION_CLAIM_CHANGED')
            if terminal_path.exists():
                terminal = reader.json(runtime.reference(terminal_path))
                _require(terminal['claim'] == runtime.reference(claim_path), 'EVALUATION_TERMINAL_CHANGED')
                return {'status': 'EVALUATION_ALREADY_CONSUMED', 'original_status': terminal['status'],
                    'terminal': runtime.reference(terminal_path), 'result_accesses_consumed': 0}
            ref = runtime.put_new(terminal_path, {'status': 'INTERRUPTED_LOOK_NO_RETRY',
                'claim': runtime.reference(claim_path), 'at': runtime.utc_now().isoformat()})
            return {'status': 'INTERRUPTED_LOOK_NO_RETRY', 'terminal': ref, 'result_accesses_consumed': 0}
        populations, date_accounting, members, by_date = _membership(reader, job, experiment_plan)
        entries = _manifest(reader, job, experiment_plan, members)
        job = {**job, 'population_by_date': by_date}
        prepared = []
        for entry in entries:
            if entry['forecast_status'] == 'SEALED_PREJUMP':
                row, payload = _forecast(reader, entry, job, experiment_plan, members[entry['race_id']])
            else:
                row, payload = _failed_forecast(reader, entry, job), None
            prepared.append((entry, row, payload))
        # Freeze the entire admitted graph after all outcome-free checks, before
        # identity proofs, official closure bytes or target projections are read.
        claim = runtime.put_new(claim_path, {'schema_version': 'prospective_speed_evaluation_claim_v1',
            'job': job_reference, 'plan': job['plan'], 'allocation': job['allocation'],
            'result_manifest': job['result_manifest'], 'evaluation_authority': job['evaluation_authority'],
            'populations': job['populations'], 'date_accounting': job['date_accounting'],
            'claimed_at': runtime.utc_now().isoformat(), 'looks_consumed': 1})
        opened = 0
        try:
            rows = []
            for entry, row, payload in prepared:
                if row['label_status'] in plan.VERIFIED_LABELS:
                    runtime.put_new(root / f'result-access-{opened + 1:03d}.json', {
                        'race_id': row['race_id'], 'target': entry['target'],
                        'identity_proof': entry['identity_proof'], 'closure_evidence': entry['closure_evidence'],
                        'claim': claim, 'at': runtime.utc_now().isoformat()})
                    opened += 1  # Consumed before the first potentially outcome-bearing read.
                    row = _target(reader, entry, row, payload, job, experiment_plan)
                rows.append(row)
            runtime.put_new(root / 'evaluation-inputs.private.json', rows)
            result = plan.evaluate_closed_population(experiment_plan, populations, rows,
                date_accounting=date_accounting, now=runtime.utc_now().isoformat(),
                collection_terminal=True, closure_terminal=True)
            result_ref = runtime.put_new(root / 'evaluation.private.json', result)
            terminal = {'status': 'COMPLETE_SINGLE_PLANNED_EVALUATION', 'claim': claim,
                'result': result_ref, 'result_accesses_consumed': opened,
                'reads': reader.reads, 'bytes': reader.bytes, 'at': runtime.utc_now().isoformat()}
        except Exception as error:
            terminal = {'status': 'FAILED_LOOK_NO_RETRY', 'claim': claim,
                'error_type': type(error).__name__, 'result_accesses_consumed': opened,
                'reads': reader.reads, 'bytes': reader.bytes, 'at': runtime.utc_now().isoformat()}
        terminal_ref = runtime.put_new(terminal_path, terminal)
        return {'status': terminal['status'], 'terminal': terminal_ref,
            'result_accesses_consumed': opened}
