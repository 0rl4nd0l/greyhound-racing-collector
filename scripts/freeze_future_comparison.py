"""Fit exactly the two #193 shortlist recipes once on the qualified 331 races."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
import time

from scripts.offline_prediction_research import scalar
from scripts.offline_systematic_search import linear_fit, serialize_model, predict as reference_predict
from src.predictor.comparison_candidates import SCHEMA, RECIPES, FEATURE_CONTRACT, card_features, predict

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_SHA256 = "c58bd59bc0d52666d31812dccd60981de7f2f03b64862f98ddf9f88ef60cec46"


def digest(raw): return hashlib.sha256(raw).hexdigest()
def encode(value): return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def freeze(prepared, out):
    wall = time.monotonic(); cpu = time.process_time()
    protection = json.loads((prepared/"protected_records.json").read_bytes())
    raw = (prepared/"development.jsonl").read_bytes()
    if digest(raw) != DEVELOPMENT_SHA256: raise ValueError("qualified development hash changed")
    for line in raw.decode().splitlines():
        if scalar(line, "race_date") > "2026-07-09": raise ValueError("ineligible date before label decode")
    rows = [json.loads(line) for line in raw.splitlines()]
    by_race = defaultdict(list)
    for row in rows: by_race[row["race_id"]].append(row)
    if len(rows) != 2360 or len(by_race) != 331: raise ValueError("qualified population changed")
    from scripts.offline_form_packet import _race_key
    if any(_race_key(rid) in protection["records"] for rid in by_race): raise ValueError("protected identity")
    # A full as-of feature reconstruction parity check, not another backtest.
    provenance = json.loads((prepared/"form_provenance.json").read_bytes())
    sources = {r["race_id"]: r for r in provenance["sources"]}
    for rid, rr in by_race.items():
        source = sources[rid]
        card = Path(source["card_source_path"]).read_bytes(); sidecar = Path(source["card_sidecar_path"]).read_bytes()
        if digest(card) != source["card_source_sha256"] or digest(sidecar) != source["card_sidecar_sha256"]:
            raise ValueError("source changed")
        metadata = json.loads(sidecar)
        from scripts.build_form_only_v1_packet import capture_timestamp
        made = card_features(card, metadata, rid, [{"box_number": r["box"], "display_name": r["dog_token"]} for r in rr],
                             captured_at=capture_timestamp(metadata, require_timezone=True))
        original = {r["box"]: r["features"] for r in rr}
        for row in made:
            if any(value != original[row["box_number"]][name] for name, value in row["features"].items()):
                raise ValueError("candidate feature parity failed")
    out.mkdir(parents=True, exist_ok=False)
    def put(name, value):
        payload = encode(value)
        with (out/name).open("xb") as f: f.write(payload)
        return digest(payload)
    identities = [{k:r[k] for k in ("race_id", "race_date", "box", "dog_token")} for r in rows]
    training_hash = put("training_identities.json", identities)
    pins = {name:digest((ROOT/name).read_bytes()) for name in (
        "scripts/freeze_future_comparison.py", "scripts/offline_form_packet.py", "scripts/build_form_only_v1_packet.py",
        "scripts/offline_systematic_search.py", "src/predictor/comparison_candidates.py")}
    fitted = {}; parity = {}
    for candidate, recipe in RECIPES.items():
        model = linear_fit(rows, recipe["features"], 1.0)
        artifact = {"schema_version":SCHEMA, "candidate_id":candidate,
                    "recipe":{**recipe,"l2":1.0,"cap":.35}, "feature_contract":FEATURE_CONTRACT,
                    "fitted":serialize_model(model), "training_population_sha256":DEVELOPMENT_SHA256,
                    "training_identities_sha256":training_hash, "source_sha256":pins}
        # Compare inference arithmetic only; no metrics or model selection.
        ref = reference_predict(rows, model, recipe["strength"])
        actual = [p for rr in by_race.values() for p in predict(artifact, rr, [r["market"] for r in rr])]
        parity[candidate] = max(abs(float(a)-b) for a,b in zip(ref,actual))
        if parity[candidate] > 1e-12: raise ValueError("candidate inference parity failed")
        fitted[candidate] = {"path":candidate+".json", "sha256":put(candidate+".json", artifact)}
    production = {name:digest((ROOT/name).read_bytes()) for name in (
        "artifacts/frozen_models/market_form_residual_v1/model.json",
        "artifacts/frozen_models/market_form_residual_v1/manifest.json", "configs/prediction/manual-default.json")}
    manifest = {"schema_version":"offline_193_candidate_registry_v1", "historical_head":"956d8289b8be062e9fe5ccdf554b9a9462391c20",
                "created_at":datetime.now(timezone.utc).isoformat(), "candidates":fitted,
                "training":{"races":331,"runners":2360,"date_min":"2026-06-10","date_max":"2026-07-08",
                            "identities_sha256":training_hash,"development_sha256":DEVELOPMENT_SHA256,
                            "protected_manifest_sha256":digest((prepared/"protected_records.json").read_bytes()),
                            "feature_reconstruction_matches":2360},
                "production":production,"source_sha256":pins,"inference_max_abs_error":parity,
                "environment":{"python":platform.python_version(),"executable":sys.executable,
                    "packages":{n:importlib.metadata.version(n) for n in ("numpy","scipy","scikit-learn")}},
                "compute":{"fits":2,"wall_seconds":time.monotonic()-wall,"cpu_seconds":time.process_time()-cpu,
                    "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
                "status":"FITTED_EXPERIMENTAL_NOT_PRODUCTION_NOT_ACTIVATED"}
    put("registry.json", manifest)
    print(json.dumps({"registry_sha256":digest((out/"registry.json").read_bytes()),"candidates":fitted,"compute":manifest["compute"]}))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();freeze(a.prepared,a.out)
