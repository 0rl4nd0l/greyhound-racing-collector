import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import build_semantic_anti_loop_seed as seed


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _fixture_artifacts(tmp_path: Path, *, complete_overlap: int = 135):
    floor = tmp_path / "floor.json"
    evaluation = tmp_path / "evaluation.json"
    overlap = tmp_path / "overlap.csv"
    bridge = tmp_path / "bridge.json"

    _write_json(
        floor,
        {
            "schema_version": "thedogs_published_market_history_audit_v1",
            "source_class": "thedogs_published_market_history",
            "complete_thedogs_published_market_history_races": 663,
            "required_floor": 300,
            "meets_300_floor": True,
            "no_write_guarantees": {
                "report_only": True,
                "db_writes": False,
                "registry_writes": False,
            },
        },
    )
    _write_json(
        evaluation,
        {
            "schema_version": (
                "thedogs_published_market_large_csv_history_challenger_report_v2"
            ),
            "final_status": "KEEP_BASELINE",
            "decision": {
                "recommendation": "KEEP_BASELINE",
                "passes_all_acceptance_gates": False,
                "qualifying_candidate_keys": [],
            },
            "floors_and_split": {"complete_races": 663, "eval_races": 300},
            "training": {"model_count": 9},
            "strict_sportsbet_baseline": {
                "status": "DATA_MISSING_FLOOR",
                "complete_eval_overlap_races": 135,
                "same_floor_overlap_cleared": False,
                "kept_separate_from_published_market_gate": True,
            },
        },
    )
    with overlap.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["race_id", "complete_strict_sportsbet_win_odds"],
        )
        writer.writeheader()
        for index in range(663):
            writer.writerow(
                {
                    "race_id": f"race-{index}",
                    "complete_strict_sportsbet_win_odds": index < complete_overlap,
                }
            )
    _write_json(
        bridge,
        {
            "schema_version": "historical_race_identity_bridge_proof_v1",
            "bridge_result": "REPORT_ONLY_BRIDGE_READY",
            "result": "COPY_REPAIR_BLOCKED",
            "writes_performed": {
                "canonical_database": False,
                "database_copy_apply": False,
                "reader_deployment": False,
                "runtime": False,
            },
            "bridge": {"race_count": 662, "snapshot_count": 1872},
            "reader_manifest": {"blockers": ["one", "two", "three"]},
        },
    )
    paths = {
        "floor_summary": floor,
        "evaluation_results": evaluation,
        "strict_overlap": overlap,
        "bridge_proof": bridge,
    }
    hashes = {
        "floor": seed.sha256_file(floor),
        "evaluation": seed.sha256_file(evaluation),
        "strict_overlap": seed.sha256_file(overlap),
        "bridge": seed.sha256_file(bridge),
    }
    return paths, hashes


def test_builds_exactly_four_hash_bound_decisions(tmp_path):
    paths, hashes = _fixture_artifacts(tmp_path)

    entries = seed.build_seed_entries(**paths, expected_hashes=hashes)

    assert len(entries) == 4
    assert [entry["decision"] for entry in entries] == [
        "PASS",
        "FAIL",
        "DATA_MISSING",
        "PASS",
    ]
    assert entries[1]["phase_after"] == "keep_market_only_baseline"
    assert entries[2]["program_track"] == "prospective_readiness"
    assert entries[2]["evidence_hash"] == (
        "sha256:" + seed.composite_evidence_hash(hashes["evaluation"], hashes["strict_overlap"])
    )
    assert entries[2]["evidence_refs"] == [
        str(paths["evaluation_results"].resolve()),
        str(paths["strict_overlap"].resolve()),
    ]
    assert "offline_research_fit" in entries[2]["does_not_block"]
    assert entries[3]["blocks"] == ["canonical_copy_repair"]
    assert all(len(entry["scope_fingerprint"]) == 64 for entry in entries)


def test_manifest_is_byte_deterministic(tmp_path):
    paths, hashes = _fixture_artifacts(tmp_path)

    first = seed.manifest_text(
        seed.build_seed_entries(**paths, expected_hashes=hashes)
    )
    second = seed.manifest_text(
        seed.build_seed_entries(**paths, expected_hashes=hashes)
    )

    assert first == second
    assert len(first.splitlines()) == 4


def test_repeat_scope_is_stable_and_changed_dataset_is_new(tmp_path):
    paths, hashes = _fixture_artifacts(tmp_path)
    floor = seed.build_seed_entries(**paths, expected_hashes=hashes)[0]

    reordered = dict(reversed(list(floor.items())))
    changed = {**floor, "dataset_version": "new_664_race_snapshot"}

    assert seed.compute_scope_fingerprint(reordered) == floor["scope_fingerprint"]
    assert seed.compute_scope_fingerprint(changed) != floor["scope_fingerprint"]


def test_rejects_tampered_pinned_artifact(tmp_path):
    paths, hashes = _fixture_artifacts(tmp_path)
    paths["floor_summary"].write_text("{}", encoding="utf-8")

    with pytest.raises(seed.SeedEvidenceError, match="SHA-256 mismatch"):
        seed.build_seed_entries(**paths, expected_hashes=hashes)


def test_rejects_wrong_strict_overlap_count(tmp_path):
    paths, hashes = _fixture_artifacts(tmp_path, complete_overlap=134)

    with pytest.raises(seed.SeedEvidenceError, match="exactly 135 complete"):
        seed.build_seed_entries(**paths, expected_hashes=hashes)


@pytest.mark.parametrize("changed_component", ["evaluation", "strict_overlap"])
def test_strict_sportsbet_composite_and_fingerprint_change_with_either_artifact(
    tmp_path: Path,
    changed_component: str,
) -> None:
    paths, hashes = _fixture_artifacts(tmp_path)
    baseline = seed.build_seed_entries(**paths, expected_hashes=hashes)[2]

    if changed_component == "evaluation":
        evaluation = json.loads(paths["evaluation_results"].read_text(encoding="utf-8"))
        evaluation["artifact_revision"] = "hash-only-test"
        _write_json(paths["evaluation_results"], evaluation)
    else:
        paths["strict_overlap"].write_text(
            paths["strict_overlap"].read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    hashes[changed_component] = seed.sha256_file(
        paths["evaluation_results"]
        if changed_component == "evaluation"
        else paths["strict_overlap"]
    )

    changed = seed.build_seed_entries(**paths, expected_hashes=hashes)[2]

    assert changed["evidence_hash"] != baseline["evidence_hash"]
    assert changed["scope_fingerprint"] != baseline["scope_fingerprint"]


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_bridge_requires_exact_no_write_key_set(tmp_path: Path, mutation: str) -> None:
    paths, hashes = _fixture_artifacts(tmp_path)
    bridge = json.loads(paths["bridge_proof"].read_text(encoding="utf-8"))
    if mutation == "missing":
        del bridge["writes_performed"]["reader_deployment"]
    else:
        bridge["writes_performed"]["unreviewed_surface"] = False
    _write_json(paths["bridge_proof"], bridge)
    hashes["bridge"] = seed.sha256_file(paths["bridge_proof"])

    with pytest.raises(seed.SeedEvidenceError, match="exactly the four no-write keys"):
        seed.build_seed_entries(**paths, expected_hashes=hashes)


def test_builder_hashes_and_parses_each_artifact_from_one_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, hashes = _fixture_artifacts(tmp_path)
    reads = {path.resolve(): 0 for path in paths.values()}
    original_read_bytes = Path.read_bytes

    def counted_read_bytes(path: Path) -> bytes:
        resolved = path.resolve()
        if resolved in reads:
            reads[resolved] += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    seed.build_seed_entries(**paths, expected_hashes=hashes)

    assert reads == {path.resolve(): 1 for path in paths.values()}


def _write_floor_task_card(path: Path, entry: dict, *, dataset_version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "---",
                f"job_id: floor-proof-{dataset_version}",
                "lane: Reporting",
                "owner: Codex",
                "allowed_files:",
                f"  - {path.relative_to(path.parents[2]).as_posix()}",
                "approval_required: true",
                "timeout_seconds: 300",
                f"output_dir: reports/agent_jobs/floor-proof-{dataset_version}",
                "mutation_mode: safe_extension",
                "production_data_access: false",
                "control_contract_version: 2",
                f"project_id: {entry['project_id']}",
                f"claim_id: {entry['claim_id']}",
                "proof_question: Does the recorded snapshot clear the declared floor?",
                f"hypothesis_id: {entry['hypothesis_id']}",
                f"program_track: {entry['program_track']}",
                f"entry_state: {entry['phase_before']}",
                f"target_transition: {entry['target_transition']}",
                "exit_predicate: The matching durable decision is reused without a report.",
                f"source_class: {entry['source_class']}",
                f"dataset_version: {dataset_version}",
                f"evidence_hash: {entry['evidence_hash']}",
                "capabilities:",
                "  - READ",
                "resume_only_if: The dataset, evidence hash, or hypothesis changes.",
                "---",
                "",
                "Focused semantic reuse fixture.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_json(command: list[str], *, cwd: Path, env: dict[str, str]) -> tuple[int, dict]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.stdout.strip(), completed.stderr
    return completed.returncode, json.loads(completed.stdout)


def test_portable_preflight_reuses_seed_and_admits_changed_dataset(
    tmp_path: Path,
) -> None:
    configured_root = os.environ.get("TENN_CONTROL_PLANE_ROOT", "").strip()
    control_root = Path(
        configured_root or Path.home() / "tenn-semantic-anti-loop-v2-canonical"
    ).expanduser()
    configured_guard = os.environ.get("TENN_GIT_GUARD_PATH", "").strip()
    guard = Path(
        configured_guard
        or Path.home()
        / ".agents/skills/tenn-git-guard/scripts/tenn_git_guard.py"
    ).expanduser()
    ledger = control_root / "scripts/agent_decision_ledger.py"
    contract = control_root / "scripts/agent_job_contract.py"
    unavailable = [path for path in (guard, ledger, contract) if not path.is_file()]
    if unavailable:
        pytest.skip("installed Tenn V2 control paths unavailable: " + ", ".join(map(str, unavailable)))

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    paths, hashes = _fixture_artifacts(artifact_dir)
    entries = seed.build_seed_entries(**paths, expected_hashes=hashes)
    floor = entries[0]

    remote = tmp_path / "greyhound-fixture.git"
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=master", str(remote)],
        check=True,
        stdout=subprocess.PIPE,
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=master"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    )
    subprocess.run(
        ["git", "config", "user.email", "semantic-seed@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Semantic Seed Test"],
        cwd=repo,
        check=True,
    )
    exact_card = repo / "docs/agent_tasks/exact-floor.md"
    changed_card = repo / "docs/agent_tasks/changed-floor.md"
    _write_floor_task_card(exact_card, floor, dataset_version=floor["dataset_version"])
    _write_floor_task_card(changed_card, floor, dataset_version="new_664_race_snapshot")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add semantic reuse fixtures"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    )
    remote_name = "greyhound-source"
    subprocess.run(
        ["git", "remote", "add", remote_name, str(remote)],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "push", "--set-upstream", remote_name, "master"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        ["git", "remote", "set-head", remote_name, "master"],
        cwd=repo,
        check=True,
    )
    topic_branch = "semantic-v2-pilot"
    subprocess.run(
        ["git", "switch", "--create", topic_branch],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    )
    subprocess.run(
        ["git", "push", "--set-upstream", remote_name, topic_branch],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    shared_registry = tmp_path / "shared-registry"
    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "TENN_AGENT_REGISTRY_ROOT": str(shared_registry),
            "TENN_CONTROL_PLANE_ROOT": str(control_root.resolve()),
        }
    )
    initialized_code, initialized = _run_json(
        [
            sys.executable,
            str(ledger),
            "initialize",
            "--repo-root",
            str(repo),
            "--authorize-create-empty-ledger",
        ],
        cwd=repo,
        env=env,
    )
    assert initialized_code == 0
    assert initialized["ok"] is True
    for entry in entries:
        appended_code, appended = _run_json(
            [
                sys.executable,
                str(ledger),
                "append",
                "--repo-root",
                str(repo),
                "--entry-json",
                json.dumps(entry, sort_keys=True),
            ],
            cwd=repo,
            env=env,
        )
        assert appended_code == 0
        assert appended["ok"] is True

    exact_code, exact = _run_json(
        [
            sys.executable,
            str(guard),
            "preflight",
            "--repo-root",
            str(repo),
            "--task-card",
            str(exact_card),
            "--json",
        ],
        cwd=repo,
        env=env,
    )
    changed_code, changed = _run_json(
        [
            sys.executable,
            str(guard),
            "preflight",
            "--repo-root",
            str(repo),
            "--task-card",
            str(changed_card),
            "--json",
        ],
        cwd=repo,
        env=env,
    )

    assert exact_code == 3, json.dumps(exact, sort_keys=True)
    assert exact["semantic_control_status"] == "REUSED_COMPLETE"
    assert exact["report_write_permitted"] is False
    assert changed_code == 0, json.dumps(changed, sort_keys=True)
    assert changed["semantic_control_status"] == "ALLOW_CHANGED_EVIDENCE"
    assert changed["substantive_work_permitted"] is True
    assert not (repo / "reports").exists()
