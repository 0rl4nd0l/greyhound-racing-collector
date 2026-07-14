from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_MANIFEST = (
    REPO_ROOT / "docs" / "agent_decisions" / "greyhound_semantic_anti_loop_seed_v1.jsonl"
)
CONTROL_ROOT = Path(
    os.environ.get(
        "TENN_CONTROL_PLANE_ROOT",
        Path.home() / "tenn-semantic-anti-loop-v2-canonical",
    )
).expanduser()
CONTRACT = CONTROL_ROOT / "scripts" / "agent_job_contract.py"
LEDGER = CONTROL_ROOT / "scripts" / "agent_decision_ledger.py"
REGISTRY = CONTROL_ROOT / "scripts" / "agent_job_registry.py"
GUARD = Path(
    os.environ.get(
        "TENN_GIT_GUARD_PATH",
        Path.home() / ".codex/skills/tenn-git-guard/scripts/tenn_git_guard.py",
    )
).expanduser()

SCOPE_FIELDS = (
    "project_id",
    "claim_id",
    "hypothesis_id",
    "source_class",
    "dataset_version",
    "evidence_hash",
    "target_transition",
)


@pytest.fixture(scope="module")
def require_installed_control_plane() -> None:
    unavailable = [path for path in (CONTRACT, LEDGER, REGISTRY, GUARD) if not path.is_file()]
    if unavailable:
        pytest.skip(
            "installed Tenn V2 control paths unavailable: "
            + ", ".join(str(path) for path in unavailable)
        )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _make_repo(path: Path) -> Path:
    remote = path.parent / f"{path.name}.git"
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=master", str(remote)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    path.mkdir()
    _git(path, "init", "--initial-branch=master")
    _git(path, "config", "user.email", "semantic-v2@example.invalid")
    _git(path, "config", "user.name", "Semantic V2 Test")
    (path / ".gitignore").write_text(
        ".tenn/\nreports/agent_jobs/\n",
        encoding="utf-8",
    )
    (path / "README.md").write_text("semantic V2 fixture\n", encoding="utf-8")
    _git(path, "add", ".gitignore", "README.md")
    _git(path, "commit", "-m", "initialize fixture")
    _git(path, "remote", "add", "origin", str(remote))
    _git(path, "push", "--set-upstream", "origin", "master")
    _git(path, "remote", "set-head", "origin", "master")
    _git(path, "switch", "--create", "semantic-v2-fixture")
    _git(path, "push", "--set-upstream", "origin", "semantic-v2-fixture")
    return path


def _env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "TENN_AGENT_REGISTRY_ROOT": str(tmp_path / "registry"),
            "TENN_AGENT_SESSION_ID": "greyhound-semantic-v2-test-session",
            "TENN_CONTROL_PLANE_ROOT": str(CONTROL_ROOT.resolve()),
        }
    )
    return env


def _run_json(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdin: dict[str, object] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        input=json.dumps(stdin) if stdin is not None else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.stdout.strip(), completed.stderr
    return completed, json.loads(completed.stdout)


def _seed_entries() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in SEED_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _scope_fingerprint(scope: dict[str, object]) -> str:
    values: list[str] = []
    for field in SCOPE_FIELDS:
        value = str(scope[field]).strip()
        if field == "evidence_hash":
            digest = value.lower().removeprefix("sha256:")
            value = f"sha256:{digest}"
        values.append(value)
    canonical = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _initialize_ledger(repo: Path, env: dict[str, str]) -> None:
    completed, payload = _run_json(
        [
            sys.executable,
            str(LEDGER),
            "initialize",
            "--repo-root",
            str(repo),
            "--authorize-create-empty-ledger",
        ],
        cwd=repo,
        env=env,
    )
    assert completed.returncode == 0, payload
    assert payload["ok"] is True


def _append_seed(
    repo: Path,
    env: dict[str, str],
    entry: dict[str, object],
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    return _run_json(
        [
            sys.executable,
            str(LEDGER),
            "append",
            "--repo-root",
            str(repo),
            "--entry-json",
            json.dumps(entry, sort_keys=True),
            "--authorize-unclaimed-seed",
        ],
        cwd=repo,
        env=env,
    )


def _write_v2_card(
    repo: Path,
    *,
    job_id: str,
    scope: dict[str, object],
    capabilities: list[str] | None = None,
    entry_state: str | None = None,
    resume_only_if: str = "The dataset, evidence hash, or hypothesis changes.",
    include_next_goal: bool = False,
) -> Path:
    output_dir = f"reports/agent_jobs/{job_id}"
    card = repo / "docs" / "agent_tasks" / f"{job_id}.md"
    card.parent.mkdir(parents=True, exist_ok=True)
    allowed_files = [
        f"{output_dir}/RUN_OUTCOME.json",
        f"{output_dir}/DECISION_ENTRY.json",
    ]
    if include_next_goal:
        allowed_files.append(f"{output_dir}/NEXT_GOAL.md")
    lines = [
        "---",
        f"job_id: {job_id}",
        "lane: Reporting",
        "owner: Codex",
        "allowed_files:",
        *(f"  - {path}" for path in allowed_files),
        "approval_required: true",
        "timeout_seconds: 300",
        f"output_dir: {output_dir}",
        "mutation_mode: safe_extension",
        "production_data_access: false",
        "control_contract_version: 2",
        f"project_id: {scope['project_id']}",
        f"claim_id: {scope['claim_id']}",
        "proof_question: Does this immutable fixture permit the requested transition?",
        f"hypothesis_id: {scope['hypothesis_id']}",
        f"program_track: {scope['program_track']}",
        f"entry_state: {entry_state or scope['phase_before']}",
        f"target_transition: {scope['target_transition']}",
        "exit_predicate: The semantic decision gate classifies this scope.",
        f"source_class: {scope['source_class']}",
        f"dataset_version: {scope['dataset_version']}",
        f"evidence_hash: {scope['evidence_hash']}",
        "capabilities:",
        *(f"  - {capability}" for capability in (capabilities or ["READ", "REPORT_WRITE"])),
        f"resume_only_if: {resume_only_if}",
        "---",
        "",
        "Semantic V2 integration fixture.",
        "",
    ]
    card.write_text("\n".join(lines), encoding="utf-8")
    return card


def _commit_cards(repo: Path, *cards: Path) -> None:
    _git(repo, "add", *(card.relative_to(repo).as_posix() for card in cards))
    _git(repo, "commit", "-m", "declare semantic V2 fixture")


def _preflight(
    repo: Path,
    card: Path,
    env: dict[str, str],
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    return _run_json(
        [
            sys.executable,
            str(GUARD),
            "preflight",
            "--repo-root",
            str(repo),
            "--task-card",
            str(card),
            "--json",
        ],
        cwd=repo,
        env=env,
    )


def _claim(
    repo: Path,
    card: Path,
    env: dict[str, str],
) -> dict[str, object]:
    completed, payload = _run_json(
        [
            sys.executable,
            str(REGISTRY),
            "claim",
            str(card),
            "--repo-root",
            str(repo),
        ],
        cwd=repo,
        env=env,
    )
    assert completed.returncode == 0, payload
    assert payload["ok"] is True
    return payload


def _write_closeout(
    repo: Path,
    *,
    card_scope: dict[str, object],
    job_id: str,
    run_id: str,
    status: str,
    decision: str,
    decision_delta: str,
    include_next_goal: bool = False,
) -> None:
    output_dir = repo / "reports" / "agent_jobs" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    state_before = str(card_scope["phase_before"])
    state_after = str(card_scope["target_transition"]) if status == "ADVANCED" else state_before
    produced_artifacts = [f"reports/agent_jobs/{job_id}/RUN_OUTCOME.json"]
    if include_next_goal:
        produced_artifacts.append(f"reports/agent_jobs/{job_id}/NEXT_GOAL.md")
    outcome = {
        "status": status,
        "scope_fingerprint": _scope_fingerprint(card_scope),
        "state_before": state_before,
        "state_after": state_after,
        "decision_delta": decision_delta,
        "reused_claims": [],
        "changed_claims": [card_scope["claim_id"]] if status == "ADVANCED" else [],
        "new_evidence": ["immutable fixture"] if status == "ADVANCED" else [],
        "produced_artifacts": produced_artifacts,
        "resume_only_if": "The dataset, evidence hash, or hypothesis changes.",
        "new_goal_permitted": False,
        "used_capabilities": ["READ", "REPORT_WRITE"],
        "blocked_by": [],
    }
    (output_dir / "RUN_OUTCOME.json").write_text(
        json.dumps(outcome, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if include_next_goal:
        (output_dir / "NEXT_GOAL.md").write_text("Repeat the same report.\n", encoding="utf-8")
    entry = {
        "decision_id": f"{job_id}-decision-v1",
        "scope_fingerprint": _scope_fingerprint(card_scope),
        "task_id": job_id,
        "run_id": run_id,
        **{field: card_scope[field] for field in SCOPE_FIELDS},
        "program_track": card_scope["program_track"],
        "phase_before": state_before,
        "phase_after": state_after,
        "decision": decision,
        "outcome_status": status,
        "decision_delta": decision_delta,
        "evidence_refs": [f"reports/agent_jobs/{job_id}/RUN_OUTCOME.json"],
        "blocks": [],
        "does_not_block": ["unrelated_transition"],
        "validated_at": "2026-07-14T12:00:00+10:00",
        "invalidation_conditions": ["The immutable fixture is disproved."],
        "reopen_conditions": ["The dataset, evidence hash, or hypothesis changes."],
    }
    (output_dir / "DECISION_ENTRY.json").write_text(
        json.dumps(entry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_repo_hooks_enforce_v2_before_tools_and_at_stop() -> None:
    settings = json.loads((REPO_ROOT / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    before = settings["hooks"]["PreToolUse"]
    stop = settings["hooks"]["Stop"]

    assert [group["matcher"] for group in before] == ["Bash|apply_patch|Edit|Write"]
    before_command = before[0]["hooks"][0]["command"]
    stop_command = stop[0]["hooks"][0]["command"]
    for command, event in ((before_command, "BeforeTool"), (stop_command, "Stop")):
        assert command.startswith("TENN_V2_REQUIRED=1 ")
        assert "$HOME/.codex/skills/tenn-git-guard/scripts/tenn_git_guard.py" in command
        assert '--repo-root "$(git rev-parse --show-toplevel)"' in command
        assert f"--event {event}" in command
        assert "$HOME/.agents/" not in command


def test_linked_worktrees_resolve_the_same_shared_decision_ledger(
    tmp_path: Path,
    require_installed_control_plane: None,
) -> None:
    repo = _make_repo(tmp_path / "repo")
    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "--detach", str(linked), "HEAD")
    env = os.environ.copy()
    env.pop("TENN_AGENT_REGISTRY_ROOT", None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull

    primary = subprocess.run(
        [sys.executable, str(LEDGER), "resolve-path", "--repo-root", str(repo)],
        cwd=repo,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    sibling = subprocess.run(
        [sys.executable, str(LEDGER), "resolve-path", "--repo-root", str(linked)],
        cwd=linked,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()

    assert primary == sibling
    assert Path(primary).name == "decision-ledger.jsonl"
    assert Path(primary).parent.name == "tenn-agent-registry"


@pytest.mark.parametrize(
    ("scope_changes", "expected_status"),
    [
        (
            {
                "dataset_version": "thedogs_published_market_history_20260714_664",
                "evidence_hash": "sha256:" + "7" * 64,
            },
            "ALLOW_CHANGED_EVIDENCE",
        ),
        (
            {"hypothesis_id": "new_floor_eligibility_hypothesis_v2"},
            "ALLOW_NEW_HYPOTHESIS",
        ),
    ],
)
def test_changed_evidence_or_new_hypothesis_is_admitted(
    tmp_path: Path,
    require_installed_control_plane: None,
    scope_changes: dict[str, object],
    expected_status: str,
) -> None:
    repo = _make_repo(tmp_path / "repo")
    env = _env(tmp_path)
    _initialize_ledger(repo, env)
    floor = _seed_entries()[0]
    appended, payload = _append_seed(repo, env, floor)
    assert appended.returncode == 0, payload
    changed = {**floor, **scope_changes}
    card = _write_v2_card(repo, job_id="changed-floor", scope=changed)
    _commit_cards(repo, card)

    completed, result = _preflight(repo, card, env)

    assert completed.returncode == 0, result
    assert result["semantic_control_status"] == expected_status
    assert result["substantive_work_permitted"] is True


def test_missing_prospective_evidence_does_not_block_offline_research_fit(
    tmp_path: Path,
    require_installed_control_plane: None,
) -> None:
    repo = _make_repo(tmp_path / "repo")
    env = _env(tmp_path)
    _initialize_ledger(repo, env)
    prospective = _seed_entries()[2]
    appended, payload = _append_seed(repo, env, prospective)
    assert appended.returncode == 0, payload
    prospective_card = _write_v2_card(
        repo,
        job_id="prospective-comparison",
        scope=prospective,
    )
    offline_scope = {
        **prospective,
        "program_track": "offline_development",
        "phase_before": "offline_feature_research_unfitted",
        "target_transition": "offline_research_fit",
    }
    offline_card = _write_v2_card(
        repo,
        job_id="offline-research-fit",
        scope=offline_scope,
        capabilities=["READ", "RESEARCH_FIT"],
    )
    _commit_cards(repo, prospective_card, offline_card)

    blocked_process, blocked = _preflight(repo, prospective_card, env)
    allowed_process, allowed = _preflight(repo, offline_card, env)

    assert blocked_process.returncode == 3, blocked
    assert blocked["semantic_control_status"] == "DATA_MISSING"
    assert blocked["substantive_work_permitted"] is False
    assert allowed_process.returncode == 0, allowed
    assert allowed["semantic_control_status"] == "ALLOW_NEW_SCOPE"
    assert allowed["substantive_work_permitted"] is True


def test_third_related_no_delta_continuation_hits_loop_guard_without_report(
    tmp_path: Path,
    require_installed_control_plane: None,
) -> None:
    repo = _make_repo(tmp_path / "repo")
    env = _env(tmp_path)
    _initialize_ledger(repo, env)
    base = _seed_entries()[2]
    for index in (1, 2):
        entry = {
            **base,
            "decision_id": f"related-no-delta-{index}",
            "task_id": f"related-no-delta-task-{index}",
            "run_id": f"related-no-delta-run-{index}",
            "target_transition": f"related_report_only_transition_{index}",
            "phase_after": "strict_same_floor_evidence_unproven",
            "validated_at": f"2026-07-14T0{index}:00:00+10:00",
        }
        entry["scope_fingerprint"] = _scope_fingerprint(entry)
        appended, payload = _append_seed(repo, env, entry)
        assert appended.returncode == 0, payload
    third_scope = {
        **base,
        "target_transition": "related_report_only_transition_3",
    }
    card = _write_v2_card(repo, job_id="third-no-delta", scope=third_scope)
    _commit_cards(repo, card)

    completed, result = _preflight(repo, card, env)

    assert completed.returncode == 3, result
    assert result["semantic_control_status"] == "LOOP_GUARD_STOP"
    assert result["semantic_control"]["no_delta_outcomes"] == 2
    assert not (repo / "reports").exists()


def test_repeated_standalone_seed_append_is_rejected(
    tmp_path: Path,
    require_installed_control_plane: None,
) -> None:
    repo = _make_repo(tmp_path / "repo")
    env = _env(tmp_path)
    _initialize_ledger(repo, env)
    floor = _seed_entries()[0]

    first, first_payload = _append_seed(repo, env, floor)
    repeated, repeated_payload = _append_seed(repo, env, floor)

    assert first.returncode == 0, first_payload
    assert repeated.returncode == 1
    assert repeated_payload["ok"] is False
    assert "duplicate" in str(repeated_payload["issues"])


def test_v1_warns_while_malformed_v2_fails(
    tmp_path: Path,
    require_installed_control_plane: None,
) -> None:
    repo = _make_repo(tmp_path / "repo")
    env = _env(tmp_path)
    legacy = repo / "docs" / "agent_tasks" / "legacy.md"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        "\n".join(
            [
                "---",
                "job_id: legacy-v1",
                "lane: Reporting",
                "owner: Codex",
                "allowed_files:",
                "  - reports/agent_jobs/legacy-v1/report.md",
                "approval_required: true",
                "timeout_seconds: 300",
                "output_dir: reports/agent_jobs/legacy-v1",
                "mutation_mode: safe_extension",
                "production_data_access: false",
                "---",
                "",
                "Legacy fixture.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    malformed_scope = _seed_entries()[0]
    malformed = _write_v2_card(repo, job_id="malformed-v2", scope=malformed_scope)
    malformed.write_text(
        "\n".join(
            line
            for line in malformed.read_text(encoding="utf-8").splitlines()
            if not line.startswith("exit_predicate:")
        )
        + "\n",
        encoding="utf-8",
    )

    legacy_process, legacy_payload = _run_json(
        [sys.executable, str(CONTRACT), "validate", str(legacy)],
        cwd=repo,
        env=env,
    )
    malformed_process, malformed_payload = _run_json(
        [sys.executable, str(CONTRACT), "validate", str(malformed)],
        cwd=repo,
        env=env,
    )

    assert legacy_process.returncode == 0, legacy_payload
    assert any("legacy v1" in warning["message"] for warning in legacy_payload["warnings"])
    assert malformed_process.returncode == 1
    assert any(issue["field"] == "exit_predicate" for issue in malformed_payload["issues"])


@pytest.mark.parametrize(
    ("status", "decision", "decision_delta", "include_next_goal", "issue_text"),
    [
        ("ADVANCED", "PASS", "", False, "decision_delta"),
        (
            "BLOCKED_NO_NEW_INPUT",
            "PARKED",
            "NO_DELTA",
            True,
            "NEXT_GOAL.md",
        ),
    ],
)
def test_invalid_v2_closeout_cannot_release_or_append_decision(
    tmp_path: Path,
    require_installed_control_plane: None,
    status: str,
    decision: str,
    decision_delta: str,
    include_next_goal: bool,
    issue_text: str,
) -> None:
    repo = _make_repo(tmp_path / "repo")
    env = _env(tmp_path)
    _initialize_ledger(repo, env)
    scope = {
        **_seed_entries()[0],
        "claim_id": "invalid_closeout_fixture",
        "hypothesis_id": "invalid_closeout_is_rejected",
        "dataset_version": "invalid_closeout_v1",
        "evidence_hash": "sha256:" + "8" * 64,
        "phase_before": "closeout_unvalidated",
        "target_transition": "closeout_validated",
    }
    job_id = "invalid-next-goal" if include_next_goal else "invalid-empty-delta"
    card = _write_v2_card(
        repo,
        job_id=job_id,
        scope=scope,
        include_next_goal=include_next_goal,
    )
    _commit_cards(repo, card)
    claim = _claim(repo, card, env)
    record = claim["record"]
    assert isinstance(record, dict)
    _write_closeout(
        repo,
        card_scope=scope,
        job_id=job_id,
        run_id=str(record["session_id"]),
        status=status,
        decision=decision,
        decision_delta=decision_delta,
        include_next_goal=include_next_goal,
    )

    released, payload = _run_json(
        [
            sys.executable,
            str(REGISTRY),
            "release",
            job_id,
            "--repo-root",
            str(repo),
        ],
        cwd=repo,
        env=env,
    )
    validated, ledger_payload = _run_json(
        [sys.executable, str(LEDGER), "validate", "--repo-root", str(repo)],
        cwd=repo,
        env=env,
    )

    assert released.returncode == 1
    assert payload["ok"] is False
    assert issue_text in str(payload["issues"])
    assert validated.returncode == 0, ledger_payload
    assert ledger_payload["entry_count"] == 0


def test_hook_blocks_no_card_runtime_command_and_claimed_capability_mismatch(
    tmp_path: Path,
    require_installed_control_plane: None,
) -> None:
    repo = _make_repo(tmp_path / "repo")
    env = _env(tmp_path)
    env["TENN_V2_REQUIRED"] = "1"
    _initialize_ledger(repo, env)
    no_card_process, no_card = _run_json(
        [
            sys.executable,
            str(GUARD),
            "hook",
            "--repo-root",
            str(repo),
            "--platform",
            "codex",
            "--event",
            "BeforeTool",
        ],
        cwd=repo,
        env={**env, "TENN_AGENT_TASK_CARD": ""},
        stdin={
            "hook_event_name": "BeforeTool",
            "tool_name": "Bash",
            "tool_input": {"command": "systemctl --user start greyhound.service"},
        },
    )
    assert no_card_process.returncode == 0, no_card
    assert no_card["decision"] == "block"

    scope = {
        **_seed_entries()[0],
        "claim_id": "report_only_capability_fixture",
        "hypothesis_id": "report_only_cannot_research_fit",
        "dataset_version": "report_only_v1",
        "evidence_hash": "sha256:" + "9" * 64,
        "phase_before": "report_only_declared",
        "target_transition": "report_only_complete",
    }
    card = _write_v2_card(repo, job_id="report-only-capability", scope=scope)
    _commit_cards(repo, card)
    _claim(repo, card, env)
    mismatch_process, mismatch = _run_json(
        [
            sys.executable,
            str(GUARD),
            "hook",
            "--repo-root",
            str(repo),
            "--platform",
            "codex",
            "--event",
            "BeforeTool",
        ],
        cwd=repo,
        env={
            **env,
            "TENN_AGENT_TASK_CARD": card.relative_to(repo).as_posix(),
        },
        stdin={
            "hook_event_name": "BeforeTool",
            "tool_name": "Bash",
            "tool_input": {"command": "python3 train_challenger.py"},
        },
    )

    assert mismatch_process.returncode == 0, mismatch
    assert mismatch["decision"] == "block"
    assert "RESEARCH_FIT" in str(mismatch["reason"])
