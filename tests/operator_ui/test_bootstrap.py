import pytest
import shutil
import hashlib
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask
from werkzeug.security import generate_password_hash

from src.operator_ui.api import install_level_1_api, register_level_1_provider
from src.operator_ui.bootstrap import CONFIG_KEY, R3_PROFILE_KEY, bind_configured_live_evidence, bind_configured_r3
from src.operator_ui.live_adapters import LiveEvidenceAdapters
from src.operator_ui.security import install_connected_mode
import src.operator_ui.bootstrap as bootstrap_module
from src.predictor.on_demand import resolve_model
from race_collection.synchronous_manual_capture import VerifiedCurrentRaceIndex
from src.operator_ui.job_store import Phase


def installed_app(tmp_path):
    app = Flask(__name__)
    app.config.update(
        TESTING=True, OPERATOR_UI_CONNECTED_MODE=True,
        OPERATOR_UI_SECRET_KEY="stable-connected-secret-" + "x" * 32,
        OPERATOR_UI_USERNAME="viewer",
        OPERATOR_UI_PASSWORD_HASH=generate_password_hash("correct horse"),
        OPERATOR_UI_LEVEL=1,
        OPERATOR_UI_AUDIT_DB_PATH=str(tmp_path / "audit.sqlite3"),
        OPERATOR_UI_JOB_DB_PATH=str(tmp_path / "jobs.sqlite3"),
        DATABASE_PATH=str(tmp_path / "canonical.sqlite3"),
        OPERATOR_UI_DEPLOYED_COMMIT="c" * 40,
        OPERATOR_UI_DEPLOYED_TREE="d" * 40,
        OPERATOR_UI_DEPLOYED_VERSION="test-v1",
    )
    install_connected_mode(app)
    assert install_level_1_api(app)
    return app


def exact_adapter():
    return object.__new__(LiveEvidenceAdapters)


def test_missing_config_is_fail_closed_and_side_effect_free(tmp_path):
    app = installed_app(tmp_path)
    assert bind_configured_live_evidence(app) is False
    assert app.extensions["operator_ui_level_1_api_providers"] == {}


def test_exact_adapter_is_bound_all_at_once_without_overview_or_audit(tmp_path):
    app, adapter = installed_app(tmp_path), exact_adapter()
    app.config[CONFIG_KEY] = adapter
    assert bind_configured_live_evidence(app) is True
    registry = app.extensions["operator_ui_level_1_api_providers"]
    assert set(registry) == {"upcoming_races", "race_detail", "recent_predictions", "prediction_detail", "collector", "corpus", "models", "system"}
    assert "overview" not in registry and "audit" not in registry
    assert registry["upcoming_races"] == adapter.upcoming


def test_unknown_partial_duplicate_and_replacement_bindings_are_denied(tmp_path):
    app = installed_app(tmp_path)
    app.config[CONFIG_KEY] = object()
    with pytest.raises(TypeError): bind_configured_live_evidence(app)
    adapter = exact_adapter()
    app.config[CONFIG_KEY] = adapter
    register_level_1_provider(app, "system", adapter.system)
    with pytest.raises(ValueError): bind_configured_live_evidence(app)
    assert set(app.extensions["operator_ui_level_1_api_providers"]) == {"system"}
    app = installed_app(tmp_path / "second"); app.config[CONFIG_KEY] = adapter
    bind_configured_live_evidence(app)
    with pytest.raises(RuntimeError): bind_configured_live_evidence(app)


def test_binding_occurs_only_when_called_never_during_requests(tmp_path):
    app, adapter = installed_app(tmp_path), exact_adapter()
    app.config[CONFIG_KEY] = adapter
    client = app.test_client()
    client.get("/operator-ui/api/v1/overview")
    assert app.extensions["operator_ui_level_1_api_providers"] == {}


def test_r3_startup_is_default_off_and_has_no_service_callback_or_path_injection(tmp_path):
    app = installed_app(tmp_path)
    assert bind_configured_r3(app) is False
    assert not any(key in app.config for key in ("OPERATOR_UI_R3_SERVICES", "OPERATOR_UI_R3_RUNTIME"))
    app.config[R3_PROFILE_KEY] = "../../arbitrary"
    with pytest.raises(ValueError, match="finite R3 profile"):
        bind_configured_r3(app)


def test_repository_profile_fails_closed_without_generated_binding(tmp_path):
    app=installed_app(tmp_path); app.config[R3_PROFILE_KEY]="repository-v1"
    with pytest.raises(RuntimeError,match="generated repository-v1 binding unavailable"):
        bootstrap_module.configure_r3_startup(app)


def repository_binding_fixture(tmp_path,monkeypatch):
    repo=tmp_path/"deployed-source"; source_root=Path(__file__).parents[2]; model=resolve_model("latest-research")
    profile_source=source_root/"configs/operator_ui/repository-v1.toml"
    copies={profile_source:repo/"configs/operator_ui/repository-v1.toml",source_root/"configs/prediction/manual-default.json":repo/"configs/prediction/manual-default.json",source_root/"scripts/predict_race_now.py":repo/"scripts/predict_race_now.py",model.model_path:repo/model.model_path.relative_to(source_root),model.manifest_path:repo/model.manifest_path.relative_to(source_root),model.schema_path:repo/model.schema_path.relative_to(source_root)}
    for source,target in copies.items():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    evidence=tmp_path/"authoritative-evidence"; producer=tmp_path/"producer"; operations=tmp_path/"operator-ui-operations"
    (evidence/"shadow_autopilot_daemon_runtime").mkdir(parents=True);(evidence/"manual_prediction_collector_requests_v1").mkdir();(producer/"artifacts/on_demand_prediction_runs").mkdir(parents=True);operations.mkdir()
    for directory in (repo,evidence,producer,operations,evidence/"manual_prediction_collector_requests_v1",producer/"artifacts/on_demand_prediction_runs"):directory.chmod(0o700)
    (evidence/"shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json").write_bytes(b"{}")
    python=tmp_path/"pinned-python";python.write_bytes(b"runtime");python.chmod(0o700)
    canonical=tmp_path/"canonical.sqlite3";canonical.write_bytes(b"canonical-read-only");canonical.chmod(0o400)
    profile_raw=(repo/"configs/operator_ui/repository-v1.toml").read_bytes()
    deployment={"source_commit":"21e7b02e60e82da9c4dbbb796ea435bc120e9862","source_tree":"2cfc75cd8a2af1a9e5da4986c969cb668b93af62","ui_version":"operator-ui-v1","profile_id":"repository-v1"}
    artifact_paths={"prediction_script":repo/"scripts/predict_race_now.py","prediction_config":repo/"configs/prediction/manual-default.json","model_artifact":repo/model.model_path.relative_to(source_root),"model_manifest":repo/model.manifest_path.relative_to(source_root),"model_schema":repo/model.schema_path.relative_to(source_root)}
    binding={"schema_version":"operator_ui_repository_binding_v1","profile_id":"repository-v1","generator":{"generator_id":"GHU-036-repository-v1-generator","schema_version":"operator_ui_repository_binding_generator_v1","version":"1"},"deployment":deployment,"profile_sha256":hashlib.sha256(profile_raw).hexdigest(),"artifacts":{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in artifact_paths.items()},"roots":{"source_root":str(repo.absolute()),"pinned_python":str(python.absolute()),"evidence_root":str(evidence.absolute()),"producer_root":str(producer.absolute()),"canonical_db":str(canonical.absolute()),"operations_root":str(operations.absolute())}}
    target=repo/"var/operator_ui/generated/repository-v1.binding.json";target.parent.mkdir(parents=True);target.write_text(json.dumps(binding),encoding="utf-8")
    monkeypatch.setattr(bootstrap_module,"_REPOSITORY_ROOT",repo)
    return repo,evidence,producer,operations,canonical


def test_repository_profile_binds_authoritative_sources_and_separate_operations_without_canonical_write(tmp_path,monkeypatch):
    repo,evidence,producer,operations,canonical=repository_binding_fixture(tmp_path,monkeypatch);before=canonical.read_bytes()
    app=Flask(__name__);app.config.update(TESTING=True,OPERATOR_UI_CONNECTED_MODE=True,OPERATOR_UI_SECRET_KEY="repository-secret-"+"x"*40,OPERATOR_UI_USERNAME="operator",OPERATOR_UI_PASSWORD_HASH=generate_password_hash("correct horse"),OPERATOR_UI_LEVEL=2,OPERATOR_UI_DEPLOYED_COMMIT="21e7b02e60e82da9c4dbbb796ea435bc120e9862",OPERATOR_UI_DEPLOYED_TREE="2cfc75cd8a2af1a9e5da4986c969cb668b93af62",OPERATOR_UI_DEPLOYED_VERSION="operator-ui-v1",OPERATOR_UI_DEPLOYED_PROFILE="repository-v1")
    app.config[R3_PROFILE_KEY]="repository-v1";assert bootstrap_module.configure_r3_startup(app) is True
    assert Path(app.config["OPERATOR_UI_AUDIT_DB_PATH"]).parent==operations and Path(app.config["DATABASE_PATH"])==canonical
    install_connected_mode(app);assert bind_configured_r3(app) is True
    worker=app.extensions["operator_ui_r3_services"].launch_once._worker
    assert worker.repository_root==repo and worker.current_index_evidence_root==evidence and worker.output_root==producer/"artifacts/on_demand_prediction_runs"
    assert worker.canonical_db==canonical and worker.collector_request_root==evidence/"manual_prediction_collector_requests_v1"
    assert canonical.read_bytes()==before and not (operations/"canonical.sqlite3").exists()


@pytest.mark.parametrize("unsafe",["missing_index","missing_model","symlink_index","symlink_python","unsafe_evidence","unsafe_producer"])
def test_repository_profile_missing_or_unsafe_authoritative_sources_fail_closed(tmp_path,monkeypatch,unsafe):
    repo,evidence,producer,_operations,_canonical=repository_binding_fixture(tmp_path,monkeypatch);index=evidence/"shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json"
    if unsafe=="missing_index":index.unlink()
    elif unsafe=="missing_model":(repo/"artifacts/frozen_models/market_form_residual_v1/model.json").unlink()
    elif unsafe=="symlink_index":index.unlink();index.symlink_to(evidence/"missing")
    elif unsafe=="symlink_python":
        binding=json.loads((repo/"var/operator_ui/generated/repository-v1.binding.json").read_text());python=Path(binding["roots"]["pinned_python"]);target=python.with_name("python-target");python.rename(target);python.symlink_to(target)
    elif unsafe=="unsafe_evidence":evidence.chmod(0o777)
    else:producer.chmod(0o777)
    app=Flask(__name__);app.config[R3_PROFILE_KEY]="repository-v1"
    with pytest.raises(RuntimeError,match="generated repository-v1 binding|fixed R3 runtime"):
        bootstrap_module.configure_r3_startup(app)


def test_finite_testing_fixture_profile_builds_real_repository_composition(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    source_root = Path(__file__).parents[2]
    model = resolve_model("latest-research")
    copies = {
        source_root / "configs/prediction/manual-default.json": repo / "configs/prediction/manual-default.json",
        source_root / "scripts/predict_race_now.py": repo / "scripts/predict_race_now.py",
        model.model_path: repo / model.model_path.relative_to(source_root),
        model.manifest_path: repo / model.manifest_path.relative_to(source_root),
        model.schema_path: repo / model.schema_path.relative_to(source_root),
    }
    for source, target in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    base = repo / "tests/operator_ui/fixtures/r3_runtime"; base.mkdir(parents=True); base.chmod(0o700)
    for directory in ("current_evidence","prediction_bundles","collector_requests","capture_evidence_a","capture_evidence_b"):
        (base / directory).mkdir(mode=0o700)
    for filename in ("canonical.sqlite3","current_index.json"):(base / filename).write_bytes(b"{}")
    monkeypatch.setattr(bootstrap_module, "_REPOSITORY_ROOT", repo)
    digest=hashlib.sha256(b"fixture-runners").hexdigest()
    race={"race_id":"race-fixture","jump_datetime":"2026-08-01T01:00:00+00:00","runner_set_sha256":digest,
          "runners":[{"box_number":1,"display_name":"ALPHA","identity":"alpha","source_native_runner_id":"dog-1"}]}
    view=VerifiedCurrentRaceIndex("collector_current_race_index_v2","run","2026-08-01T00:00:00Z",digest,b"{}",(race,),"source.json",digest,digest,digest,digest)
    monkeypatch.setattr(bootstrap_module,"bounded_current_race_index",lambda **_:view)
    def terminal_runner(store,job_id,_worker,*,now,confirm_audit):
        job,attempt=store.claim_attempt(job_id,now=now(),confirm_audit=confirm_audit)
        store.transition(job_id,Phase.ATTEMPT_STARTED,now=now(),status="RUNNING",reason="predictor_started",facts={"attempt_id":attempt,"pid":123},confirm_audit=confirm_audit)
        empty=hashlib.sha256(b"").hexdigest()
        facts={"attempt_id":attempt,"pid":123,"exit_code":-1,"stdout_complete":False,"stdout_prefix_length":0,"stdout_prefix_sha256":empty,"stderr_complete":False,"stderr_prefix_length":0,"stderr_prefix_sha256":empty}
        return store.transition(job_id,Phase.FAILED,now=now(),status="FAILED",reason="POST_SPAWN_FAILURE",facts=facts,confirm_audit=confirm_audit)
    monkeypatch.setattr(bootstrap_module,"run_once",terminal_runner)
    app = Flask(__name__)
    app.config.update(TESTING=True, OPERATOR_UI_CONNECTED_MODE=True,
        OPERATOR_UI_SECRET_KEY="fixture-secret-"+"x"*40, OPERATOR_UI_USERNAME="viewer",
        OPERATOR_UI_PASSWORD_HASH=generate_password_hash("correct horse"), OPERATOR_UI_LEVEL=2,
        OPERATOR_UI_AUDIT_DB_PATH=str(base/"audit.sqlite3"), DATABASE_PATH=str(base/"canonical.sqlite3"),
        OPERATOR_UI_DEPLOYED_COMMIT="c"*40, OPERATOR_UI_DEPLOYED_TREE="d"*40,
        OPERATOR_UI_DEPLOYED_VERSION="fixture")
    app.config[R3_PROFILE_KEY]="fixture-v1"
    install_connected_mode(app)
    assert bind_configured_r3(app) is True
    assert "operator_ui_r3_submit" in app.view_functions
    services = app.extensions["operator_ui_r3_services"]
    assert services.job_store.path == (base/"jobs.sqlite3").absolute()
    client=app.test_client()
    token=client.get("/operator-ui/login",base_url="https://localhost").get_json()["csrf_token"]
    token=client.post("/operator-ui/login",base_url="https://localhost",data={"username":"viewer","password":"correct horse","csrf_token":token}).get_json()["csrf_token"]
    response=client.post("/operator-ui/api/v1/prediction-jobs",base_url="https://localhost",headers={"X-CSRF-Token":token},json={"race_id":"race-fixture","model_id":"latest-research","config_id":"manual-default","odds_source_id":"auto","idempotency_key":"12345678-1234-4123-8123-123456789abc"})
    assert response.status_code==202 and response.get_json()["phase"]=="WAITING_FOR_CLAIM",response.get_json()
    job_id=response.get_json()["job_id"]
    deadline=time.monotonic()+2
    while services.job_store.get(job_id).phase is not Phase.FAILED and time.monotonic()<deadline:time.sleep(.01)
    job=services.job_store.get(job_id)
    assert job.attempt_claimed and job.phase is Phase.FAILED
    assert [event["phase"] for event in services.job_store.events(job_id)][-3:]==["CLAIMED","ATTEMPT_STARTED","FAILED"]


@pytest.mark.parametrize("unsafe_kind", ["symlink", "directory"])
def test_finite_profile_rejects_unsafe_preexisting_job_store(tmp_path, monkeypatch, unsafe_kind):
    repo=tmp_path/"repo"; source_root=Path(__file__).parents[2]; model=resolve_model("latest-research")
    for source,relative in ((source_root/"configs/prediction/manual-default.json","configs/prediction/manual-default.json"),(source_root/"scripts/predict_race_now.py","scripts/predict_race_now.py"),(model.model_path,str(model.model_path.relative_to(source_root))),(model.manifest_path,str(model.manifest_path.relative_to(source_root))),(model.schema_path,str(model.schema_path.relative_to(source_root)))):
        target=repo/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    base=repo/"tests/operator_ui/fixtures/r3_runtime"; base.mkdir(parents=True); base.chmod(0o700)
    for directory in ("current_evidence","prediction_bundles","collector_requests","capture_evidence_a","capture_evidence_b"):(base/directory).mkdir(mode=0o700)
    for filename in ("audit.sqlite3","canonical.sqlite3","current_index.json"):(base/filename).write_bytes(b"{}")
    if unsafe_kind=="symlink":(base/"jobs.sqlite3").symlink_to(base/"canonical.sqlite3")
    else:(base/"jobs.sqlite3").mkdir()
    monkeypatch.setattr(bootstrap_module,"_REPOSITORY_ROOT",repo)
    app=installed_app(tmp_path/"app"); app.config.update(TESTING=True,OPERATOR_UI_AUDIT_DB_PATH=str(base/"audit.sqlite3"),DATABASE_PATH=str(base/"canonical.sqlite3")); app.config[R3_PROFILE_KEY]="fixture-v1"
    with pytest.raises(RuntimeError,match="job store unsafe"):bind_configured_r3(app)
