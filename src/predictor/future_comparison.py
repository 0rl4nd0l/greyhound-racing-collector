"""Default-off, write-once comparison inside an existing retained prediction job.

No acquisition, scheduling, model selection, outcomes or production routing.
Scientific admission is separate from operational prediction success.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import resource
import sqlite3
import time

from src.predictor.on_demand import PredictionBlocked, canonical_bytes, sha256_bytes, write_exact_bytes

MODELS = ("market", "production", "residual_box", "residual_half")
PLAN_SCHEMA = "frozen_four_way_comparison_plan_v1"


def stamp(value):
    value = datetime.fromisoformat(value)
    if value.tzinfo is None or value.utcoffset() is None: raise ValueError("timezone_required")
    return value


def checked(path, expected):
    if path.is_symlink() or not path.is_file(): raise ValueError("comparison_file_unavailable")
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected: raise ValueError("comparison_file_hash_changed")
    return raw


def put(path, value):
    write_exact_bytes(path, canonical_bytes(value))


def load_plan(path, expected):
    raw = checked(path, expected); plan = json.loads(raw)
    if plan["schema_version"] != PLAN_SCHEMA: raise ValueError("comparison_plan_schema")
    if plan["status"] not in {"AUTHORIZED", "SYNTHETIC_REHEARSAL_ONLY"}:
        raise ValueError("comparison_not_activated")
    if not plan["authority_reference"] or plan["decision_seconds_before_jump"] != 120 or plan["quote_lead_seconds"] != [120,600]:
        raise ValueError("comparison_policy_changed")
    if set(plan["candidate_registry"]) != {"path", "sha256"}: raise ValueError("comparison_registry_binding")
    if not any(start <= "2026-07-15" and end >= "2026-09-30" for start,end in plan["denied_history_intervals"]):
        raise ValueError("historical_reservations_not_protected")
    if not stamp(plan["activated_at"]) < stamp(plan["starts_at"]) < stamp(plan["ends_at"]):
        raise ValueError("comparison_population_interval")
    if plan["status"] == "AUTHORIZED" and (not plan.get("exclusive_population_allocation_reference") or not plan.get("reservation_review_sha256") or not plan.get("machine_history_authority_reference")):
        raise ValueError("comparison_reservation_allocation_missing")
    return plan, raw


def observe_worker_job(job, config, now):
    """Consume research admission before worker preflight; production is separate."""
    try:
        plan,_=load_plan(config.comparison_plan,config.comparison_plan_sha256)
        jump=stamp(job.input.jump_timestamp)
        if not stamp(plan["starts_at"])<=jump<stamp(plan["ends_at"]) or now<stamp(plan["activated_at"]) or now>=jump-timedelta(seconds=120):
            return None
        if plan.get("race_ids") is not None and job.input.race_id not in plan["race_ids"]:
            return None
        programme=Path(plan["programme_root"])/config.comparison_plan_sha256
        if not programme.is_absolute(): return None
        key=hashlib.sha256(job.input.race_id.encode()).hexdigest()
        path=programme/"opportunities"/(key+".json")
        if not path.exists():
            try: put(path,{"race_id":job.input.race_id,"jump_timestamp":job.input.jump_timestamp,"first_observed_at":now.isoformat(),"plan_sha256":config.comparison_plan_sha256})
            except (FileExistsError, PredictionBlocked) as exc:
                if isinstance(exc,PredictionBlocked) and exc.code!="WRITE_TARGET_EXISTS": raise
        claim=programme/"attempts"/key
        claim.mkdir(parents=True,exist_ok=False)
        put(claim/"dispatch.json",{"job_id":job.job_id,"race_id":job.input.race_id,"jump_timestamp":job.input.jump_timestamp,
            "dispatched_at":now.isoformat(),"plan_sha256":config.comparison_plan_sha256,
            "retained_input_manifest_sha256":job.input.retained_input_manifest_sha256})
        return claim
    except (ValueError,KeyError,FileExistsError):
        return None  # The subprocess records the corresponding comparison failure.


def observe_verified_index(binding, view, now):
    """Record the existing verified schedule before capture/retention success.

    Called by the existing supervisor. No provider or outcome access; no claim
    that the observed index exhausts a venue's actual schedule.
    """
    plan,_=load_plan(Path(binding['path']),binding['sha256'])
    if now<stamp(plan['activated_at']): return
    root=Path(plan['programme_root'])/binding['sha256']
    for race in view.races:
        jump=stamp(race['jump_datetime'])
        if not stamp(plan['starts_at'])<=jump<stamp(plan['ends_at']): continue
        if plan.get('race_ids') is not None and race['race_id'] not in plan['race_ids']: continue
        key=hashlib.sha256(race['race_id'].encode()).hexdigest()
        path=root/'opportunities'/(key+'.json')
        if path.exists(): continue
        value={'race_id':race['race_id'],'jump_timestamp':race['jump_datetime'],'first_observed_at':now.isoformat(),
            'plan_sha256':binding['sha256'],'observed_before_decision':now<jump-timedelta(seconds=120),
            'source_generated_at':view.source_generated_at}
        try: put(path,value)
        except (FileExistsError,PredictionBlocked) as exc:
            if isinstance(exc,PredictionBlocked) and exc.code!='WRITE_TARGET_EXISTS': raise


class Comparison:
    def __init__(self, *, state, plan_path, plan_sha256, now, repository_root, retained_digest, odds_source, opportunities=()):
        self.state=state; self.bundle=Path(state["bundle"]); self.root=self.bundle/"comparison"
        self.now=now; self.repository_root=repository_root; self.claim=None; self.plan=None
        self.failed=None; self.finished=False; self.started=now(); self.wall=time.monotonic(); self.cpu=time.process_time()
        self.rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.binding={"prediction_id":state["prediction_id"], "job_id":state["job_id"], "race":state["race"],
            "runner_set_sha256":state["runner_set_sha256"], "plan_sha256":plan_sha256,
            "retained_input_manifest_sha256":retained_digest}
        try:
            self.plan, raw=load_plan(Path(plan_path),plan_sha256)
            write_exact_bytes(self.root/"plan.json",raw)
            programme=Path(self.plan["programme_root"])
            if not programme.is_absolute(): raise ValueError("comparison_programme_root_not_absolute")
            programme=programme/plan_sha256
            for opportunity in opportunities:
                off=stamp(opportunity["jump_timestamp"])
                if not stamp(self.plan["starts_at"]) <= off < stamp(self.plan["ends_at"]): continue
                if self.plan.get("race_ids") is not None and opportunity["race_id"] not in self.plan["race_ids"]: continue
                if self.started>=off-timedelta(seconds=120): continue
                key=hashlib.sha256(opportunity["race_id"].encode()).hexdigest()
                try:
                    put(programme/"opportunities"/(key+".json"),{**opportunity,"first_observed_at":self.started.isoformat(),"plan_sha256":plan_sha256})
                except (FileExistsError, PredictionBlocked) as exc:
                    if isinstance(exc,PredictionBlocked) and exc.code!="WRITE_TARGET_EXISTS": raise
                    prior=json.loads((programme/"opportunities"/(key+".json")).read_bytes())
                    if any(prior[k]!=opportunity[k] for k in ("race_id","jump_timestamp")):
                        raise ValueError("comparison_observed_race_changed")
            jump=stamp(state["race"]["jump_timestamp"])
            self.decision=jump-timedelta(seconds=120)
            if not stamp(self.plan["starts_at"]) <= jump < stamp(self.plan["ends_at"]) or self.started < stamp(self.plan["activated_at"]):
                raise ValueError("comparison_outside_population")
            if self.plan.get("race_ids") is not None and state["race"]["race_id"] not in self.plan["race_ids"]:
                raise ValueError("comparison_race_not_admitted")
            if self.started >= self.decision: raise ValueError("comparison_late_admission")
            if not retained_digest or odds_source != "receipt": raise ValueError("comparison_requires_retained_receipt")
            if state["model"].resolved != "market_form_residual_v1": raise ValueError("comparison_requires_production_predictor")
            # One consumed attempt per programme/race, even after failure/crash.
            key=hashlib.sha256(state["race"]["race_id"].encode()).hexdigest()
            self.claim=programme/"attempts"/key
            try: self.claim.mkdir(parents=True,exist_ok=False)
            except FileExistsError:
                dispatch=self.claim/"dispatch.json"
                if (self.claim/"admission.json").exists() or (self.claim/"worker_failure.json").exists() or not dispatch.exists(): raise
                if json.loads(dispatch.read_bytes())["job_id"]!=state["job_id"]: raise
            put(self.claim/"admission.json",{**self.binding,"admitted_at":self.started.isoformat(),"bundle_directory":self.bundle.name,
                "decision_at":self.decision.isoformat(),"evidence_class":self.plan["status"],"membership_before_outcomes":True})
            self.binding["admission_sha256"]=sha256_bytes((self.claim/"admission.json").read_bytes())
            registry_path=Path(self.plan["candidate_registry"]["path"])
            if not registry_path.is_absolute(): registry_path=Path(plan_path).parent/registry_path
            self.registry_path=registry_path
            registry_raw=checked(registry_path,self.plan["candidate_registry"]["sha256"])
            self.registry=json.loads(registry_raw)
            write_exact_bytes(self.root/"registry.json",registry_raw)
            if set(self.registry["candidates"]) != {"residual_box","residual_half"}:
                raise ValueError("comparison_candidate_set_changed")
            for name,digest in self.registry["production"].items():
                checked(repository_root/name,digest)
            if self.registry["production"]["configs/prediction/manual-default.json"] != state["config_sha"]:
                raise ValueError("comparison_production_config_changed")
            for name in ("scripts/build_form_only_v1_packet.py","scripts/offline_form_packet.py","src/predictor/comparison_candidates.py"):
                checked(repository_root/name,self.registry["source_sha256"][name])
        except FileExistsError:
            self.claim=None; self.failed="comparison_attempt_already_consumed"
        except Exception as exc:
            self.failed=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
        put(self.root/"attempt.json",{**self.binding,"started_at":self.started.isoformat(),"admission_failure":self.failed})
        self.admission_cpu=time.process_time()-self.cpu; self.admission_wall=time.monotonic()-self.wall

    def finish(self, *, prediction=None, receipt=None, form=None, sidecar=None, production_failure=None):
        if self.finished: return
        wall=time.monotonic(); cpu=time.process_time(); rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        common={**self.binding,"input_identity":None}
        shared_failure=self.failed or production_failure
        data=None
        if shared_failure is None:
            try:
                if self.now() >= self.decision: raise ValueError("comparison_decision_cutoff_passed")
                capture=stamp(receipt["captured_at"]); jump=stamp(self.state["race"]["jump_timestamp"])
                if not 120 <= (jump-capture).total_seconds() <= 600 or capture > self.started:
                    raise ValueError("comparison_quote_outside_decision_window")
                artifact=prediction["artifact_prediction"]
                raw=form.read_bytes(); metadata_raw=sidecar.read_bytes()
                identity={"form_sha256":sha256_bytes(raw),"sidecar_sha256":sha256_bytes(metadata_raw),
                    "odds_receipt_sha256":sha256_bytes(canonical_bytes(receipt)),
                    "capture_sha256":sha256_bytes((self.bundle/"source/capture.json").read_bytes()),
                    "production_feature_rows_sha256":artifact["input_hashes"]["feature_rows_sha256"],
                    "captured_at":capture.isoformat()}
                for ours,theirs in (("form_sha256","form_csv_sha256"),("sidecar_sha256","sidecar_sha256"),("capture_sha256","capture_artifact_sha256")):
                    if identity[ours] != artifact["input_hashes"][theirs]: raise ValueError("comparison_shared_input_mismatch")
                expected={r["box_number"]:r for r in self.state["runners"]}
                prices={r["box_number"]:r for r in receipt["markets"]["win"]}
                prod={r["box_number"]:r for r in prediction["predictions"]}
                if set(prices) != set(expected) or set(prod) != set(expected) or len(prod)!=len(prediction["predictions"]) or len(prices)!=len(receipt["markets"]["win"]):
                    raise ValueError("comparison_runner_mismatch")
                from scripts.build_form_only_v1_packet import dog_token, capture_timestamp
                boxes=sorted(expected)
                for box in boxes:
                    if any(dog_token(row["dog_name"]) != dog_token(expected[box]["display_name"]) for row in (prod[box],prices[box])) or prod[box]["win_odds"]!=prices[box]["odds_decimal"]:
                        raise ValueError("comparison_runner_or_price_mismatch")
                inverse=[1/float(prices[b]["odds_decimal"]) for b in boxes]
                market=[v/math.fsum(inverse) for v in inverse]
                if any(abs(p-prod[b]["market_probability"])>1e-12 for p,b in zip(market,boxes)):
                    raise ValueError("comparison_market_mismatch")
                common["input_identity"]=identity
                put(self.root/"inputs.json",{**identity,"runners":[{**expected[b],"win_odds":prices[b]["odds_decimal"]} for b in boxes]})
                data=(boxes,expected,market,[prod[b]["probability"] for b in boxes],raw,json.loads(metadata_raw))
            except Exception as exc:
                shared_failure=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
        features=None; feature_error=None
        if data is not None:
            try:
                # No protected results may enter through the production history
                # either. This query projects metadata only, never dog outcomes.
                denied=self.plan["denied_history_intervals"]
                uri=f"file:{self.bundle/'features/sealed_history.db'}?mode=ro&immutable=1"
                with sqlite3.connect(uri,uri=True) as conn:
                    for (day,) in conn.execute("SELECT DISTINCT race_date FROM race_metadata"):
                        if any(start<=str(day)[:10]<=end for start,end in denied):
                            raise ValueError("protected_history_in_shared_production_inputs")
                from src.predictor.comparison_candidates import card_features
                from scripts.build_form_only_v1_packet import capture_timestamp
                features=card_features(data[4],data[5],self.state["race"]["race_id"],self.state["runners"],
                    captured_at=capture_timestamp(data[5],require_timezone=True),denied_history_intervals=denied)
                put(self.root/"features.json",features)
            except Exception as exc:
                feature_error=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
        # Feature failure does not erase valid market/production outputs. A
        # protected-history error excludes the entire paired scientific record.
        records={}
        for name in MODELS:
            failure=shared_failure; probabilities=None
            model_sha=(self.state["model"].model_sha256 if name=="production" else
                getattr(self,"registry",{}).get("candidates",{}).get(name,{}).get("sha256"))
            try:
                if failure is None:
                    if name=="market": probabilities=data[2]
                    elif name=="production":
                        probabilities=data[3]; model_sha=self.state["model"].model_sha256
                    else:
                        if feature_error: raise ValueError(feature_error)
                        entry=self.registry["candidates"][name]; model_sha=entry["sha256"]
                        path=Path(entry["path"])
                        if path.is_absolute() or ".." in path.parts: raise ValueError("candidate_path_invalid")
                        raw=checked(self.registry_path.parent/path,model_sha)
                        write_exact_bytes(self.root/"artifacts"/(name+".json"),raw)
                        from src.predictor.comparison_candidates import predict
                        model=json.loads(raw)
                        if model["candidate_id"] != name: raise ValueError("candidate_identity_changed")
                        probabilities=predict(model,features,data[2])
                    if len(probabilities)!=len(data[0]) or any(not math.isfinite(p) or not 0<p<1 for p in probabilities) or abs(math.fsum(probabilities)-1)>1e-12:
                        raise ValueError("candidate_probability_invalid")
                    if self.now() >= self.decision: raise ValueError("candidate_completed_after_cutoff")
            except Exception as exc:
                failure=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
            rows=None if failure else [{"box_number":b,"identity":data[1][b]["identity"],"dog_name":data[1][b]["display_name"],"probability":p} for b,p in zip(data[0],probabilities)]
            record={**common,"candidate":name,"status":"FAILED" if failure else "SEALED", "failure":failure,
                "model_sha256":model_sha,"predictions":rows,"completed_at":self.now().isoformat()}
            put(self.root/(name+".json"),record); records[name]=record["status"]
        self.records=records
        put(self.root/"summary.json",{**self.binding,"models":records,"shared_failure":shared_failure,
            "feature_failure":feature_error,"target_outcomes_accessed":False,"production_modified":False,
            "comparison_cpu_seconds":self.admission_cpu+time.process_time()-cpu,
            "comparison_elapsed_seconds":self.admission_wall+time.monotonic()-wall,
            "score_phase_peak_rss_increment_kib":max(0,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-rss),
            "process_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        self.finished=True

    def close(self, entry):
        """Bind actual bundle publication completion, not a supplied score clock."""
        if self.claim is None: return
        completed=self.now()
        put(self.claim/"completion.json",{**self.binding,"bundle_entry":entry,"published_complete_at":completed.isoformat(),
            "decision_at":self.decision.isoformat(), "status":"COMPLETE_BEFORE_CUTOFF" if completed<self.decision else "LATE_NOT_EVALUABLE",
            "models":self.records,"evidence_class":self.plan["status"]})


def verify_comparison(root, admission_path, *, expected_plan_sha256=None):
    """Read-only verification; also works when production or a candidate failed."""
    from src.predictor.on_demand import verify_indexed_prediction_bundle
    admission=json.loads(admission_path.read_bytes())
    if expected_plan_sha256 is not None and admission["plan_sha256"]!=expected_plan_sha256:
        raise ValueError("comparison_unexpected_programme")
    completion=json.loads(admission_path.with_name("completion.json").read_bytes())
    if completion["admission_sha256"]!=sha256_bytes(admission_path.read_bytes()):
        raise ValueError("comparison_admission_changed")
    if stamp(admission["decision_at"])!=stamp(admission["race"]["jump_timestamp"])-timedelta(seconds=120) or stamp(admission["admitted_at"])>=stamp(admission["decision_at"]):
        raise ValueError("comparison_admission_timing")
    if any(completion[k]!=admission[k] for k in ("race","runner_set_sha256","plan_sha256","prediction_id","retained_input_manifest_sha256")):
        raise ValueError("comparison_completion_binding")
    verified=verify_indexed_prediction_bundle(root,completion["bundle_entry"])
    directory=root/verified.directory
    contents={name:checked(directory/name,entry["sha256"]) for name,entry in verified.manifest["files"].items()}
    if verified.result["prediction_id"]!=admission["prediction_id"] or verified.result["race"]!=admission["race"]:
        raise ValueError("comparison_bundle_binding")
    plan=json.loads(contents["comparison/plan.json"])
    if sha256_bytes(contents["comparison/plan.json"])!=admission["plan_sha256"]:
        raise ValueError("comparison_plan_changed")
    records={name:json.loads(contents[f"comparison/{name}.json"]) for name in MODELS}
    registry=json.loads(contents.get("comparison/registry.json",b"{}"))
    if registry and sha256_bytes(contents["comparison/registry.json"])!=plan["candidate_registry"]["sha256"]:
        raise ValueError("comparison_registry_changed")
    identity=None; common=None
    for name,record in records.items():
        if record["candidate"]!=name or any(record[k]!=admission[k] for k in ("race","runner_set_sha256","plan_sha256","prediction_id","retained_input_manifest_sha256")):
            raise ValueError("comparison_record_binding")
        if record["admission_sha256"]!=completion["admission_sha256"]: raise ValueError("comparison_admission_changed")
        if record["status"]=="FAILED":
            if record["predictions"] is not None or not record["failure"]: raise ValueError("comparison_failure_invalid")
            continue
        if record["status"]!="SEALED" or record["failure"] is not None or stamp(record["completed_at"])>=stamp(admission["decision_at"]):
            raise ValueError("comparison_record_timing_or_status")
        rows=record["predictions"]; keys=[(r["box_number"],r["identity"]) for r in rows]
        if len(set(keys))!=len(keys) or any(not 0<r["probability"]<1 for r in rows) or abs(math.fsum(r["probability"] for r in rows)-1)>1e-12:
            raise ValueError("comparison_normalization")
        if common is not None and (keys!=common or record["input_identity"]!=identity): raise ValueError("comparison_common_inputs")
        common=keys; identity=record["input_identity"]
    summary=json.loads(contents["comparison/summary.json"])
    if completion["models"]!={n:r["status"] for n,r in records.items()} or summary["models"]!=completion["models"]:
        raise ValueError("comparison_completion_status")
    if identity is not None:
        inputs=json.loads(contents["comparison/inputs.json"])
        for name in ("form_sha256","sidecar_sha256","odds_receipt_sha256","capture_sha256","production_feature_rows_sha256","captured_at"):
            if inputs[name]!=identity[name]: raise ValueError("comparison_input_identity")
        if inputs["odds_receipt_sha256"]!=sha256_bytes(contents["odds_receipt.json"]) or inputs["capture_sha256"]!=sha256_bytes(contents["source/capture.json"]):
            raise ValueError("comparison_snapshot_changed")
        receipt=json.loads(contents["odds_receipt.json"])
        expected=sorted((r["box_number"],r["identity"]) for r in receipt["markets"]["win"])
        if common!=expected: raise ValueError("comparison_incomplete_field")
        prices=sorted(receipt["markets"]["win"],key=lambda r:r["box_number"])
        inverse=[1/r["odds_decimal"] for r in prices]; market=[v/math.fsum(inverse) for v in inverse]
        if not 120 <= (stamp(admission["race"]["jump_timestamp"])-stamp(inputs["captured_at"])).total_seconds() <= 600:
            raise ValueError("comparison_quote_timing")
        forms=[p for p in contents if p.startswith("source/") and p.endswith(".csv")]
        if len(forms)!=1 or inputs["form_sha256"]!=sha256_bytes(contents[forms[0]]) or inputs["sidecar_sha256"]!=sha256_bytes(contents[forms[0]+".metadata.json"]):
            raise ValueError("comparison_form_changed")
        if inputs["production_feature_rows_sha256"]!=sha256_bytes(contents["features/sealed/shadow_feature_rows.json"]):
            raise ValueError("comparison_production_features_changed")
        from src.predictor.comparison_candidates import card_features
        from scripts.build_form_only_v1_packet import capture_timestamp
        metadata=json.loads(contents[forms[0]+".metadata.json"])
        if "comparison/features.json" in contents:
            replay_features=card_features(contents[forms[0]],metadata,admission["race"]["race_id"],inputs["runners"],
                captured_at=capture_timestamp(metadata,require_timezone=True),denied_history_intervals=plan["denied_history_intervals"])
            if canonical_bytes(replay_features)!=contents["comparison/features.json"]:
                raise ValueError("comparison_feature_replay_changed")
        for name,record in records.items():
            if record["status"]!="SEALED": continue
            actual=[r["probability"] for r in record["predictions"]]
            if name=="market": replay=market
            elif name=="production":
                production=sorted(verified.result["prediction"]["predictions"],key=lambda r:r["box_number"])
                replay=[r["probability"] for r in production]
                if record["model_sha256"]!=verified.result["model"]["artifact_sha256"]: raise ValueError("comparison_production_changed")
                from scripts.predict_market_form_residual import score_from_artifacts
                artifact=score_from_artifacts(race_id=admission["race"]["race_id"],form_csv_path=directory/forms[0],
                    sidecar_path=directory/(forms[0]+".metadata.json"),feature_rows_path=directory/"features/sealed/shadow_feature_rows.json",
                    feature_manifest_path=directory/"features/sealed/shadow_manifest.json",implementation_manifest_path=directory/"features/sealed/implementation_file_manifest.json",
                    capture_path=directory/"source/capture.json",model_path=directory/"model/model.json",manifest_path=directory/"model/manifest.json",
                    score_timestamp=stamp(record["completed_at"]))
                reproduced=[r["full_probability"] for r in sorted(artifact["predictions"],key=lambda r:r["box"])]
                if any(abs(a-b)>1e-12 for a,b in zip(replay,reproduced)): raise ValueError("production_replay_changed")
            else:
                raw=contents[f"comparison/artifacts/{name}.json"]
                if record["model_sha256"]!=sha256_bytes(raw) or record["model_sha256"]!=registry["candidates"][name]["sha256"]:
                    raise ValueError("comparison_candidate_changed")
                from src.predictor.comparison_candidates import predict
                replay=predict(json.loads(raw),json.loads(contents["comparison/features.json"]),market)
            if any(abs(a-b)>1e-12 for a,b in zip(actual,replay)):
                raise ValueError("comparison_replay_changed")
    eligible=(completion["status"]=="COMPLETE_BEFORE_CUTOFF" and stamp(completion["published_complete_at"])<stamp(admission["decision_at"])
        and all(r["status"]=="SEALED" for r in records.values()) and summary["feature_failure"] is None)
    return {"schema_version":"verified_four_way_comparison_v1","evidence_class":plan["status"],
        "eligible_common_race":eligible,"future_race_evidence":eligible and plan["status"]=="AUTHORIZED",
        "records":records,"completion":completion}
