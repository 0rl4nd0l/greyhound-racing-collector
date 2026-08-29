#!/usr/bin/env python3
"""Frozen report-only form/speed residual experiment against Sportsbet WIN.

The two-phase interface is intentional: ``--freeze`` builds the source-bound
feature matrix and protocol without fitting; ``--evaluate`` consumes only that
sealed matrix.  No August outcome source is opened by this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import subprocess
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector")
MATRIX = ROOT / "artifacts/sportsbet_win_market_surface_audit_20260815_report_only/canonical_win_matrix.jsonl"
TARGET = SOURCE_ROOT / "artifacts/sportsbet_speed_context_experiment_20260815_clean_rerun_report_only/enriched_development_matrix.jsonl"
HISTORY = SOURCE_ROOT / "artifacts/raw_race_shape_experiment_20260815_report_only/deduplicated_raw_history_sidecar.jsonl"
DB = SOURCE_ROOT / ".scratch/development_source_20260802.db"
BETFAIR = ROOT / "artifacts/betfair_historical_surface_20260817_report_only/sportsbet_betfair_joined_surface.jsonl"
DEFAULT_OUT = ROOT / "artifacts/form_speed_market_residual_20260818_report_only"
START, END = "2026-06-10", "2026-07-18"
FORWARD_OUTCOME_START = "2026-08-18"
FOLDS = (
    {"id": 1, "train_start": START, "train_end": "2026-06-24", "test_start": "2026-06-25", "test_end": "2026-07-02"},
    {"id": 2, "train_start": START, "train_end": "2026-07-02", "test_start": "2026-07-03", "test_end": "2026-07-10"},
    {"id": 3, "train_start": START, "train_end": "2026-07-10", "test_start": "2026-07-11", "test_end": END},
)
SEED, BOOTSTRAPS, ROI_BOOTSTRAPS, L2 = 20260818, 5000, 5000, 1.0
BASE_FEATURES = (
    "speed_median_5", "speed_best_5", "speed_trend_5", "speed_consistency_5",
    "early_pace_median_5", "early_pace_best_5", "same_track_distance_speed_mean",
    "same_track_distance_starts", "days_since_run", "finish_median_5",
    "finish_best_5", "box_position", "speed_field_rank", "speed_field_gap",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def slope(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def exact_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(row["race_id"]), int(row["box_number"]), str(row["odds_capture_timestamp"])


def parse_source_timestamp(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit("canonical_population_invalid:invalid_timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SystemExit("canonical_population_invalid:timestamp_without_timezone")
    return parsed


def validate_canonical_population(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    for race_id, rows in grouped.items():
        prefix = f"canonical_population_invalid:{race_id}:"
        if not rows:
            raise SystemExit(prefix + "empty_race")
        if any(str(row.get("race_date") or "") >= FORWARD_OUTCOME_START for row in rows):
            raise SystemExit(prefix + "forward_target_date")
        if sum(row.get("label_is_winner") == 1 for row in rows) != 1 or any(
            row.get("label_is_winner") not in (0, 1) for row in rows
        ):
            raise SystemExit(prefix + "winner_count")
        try:
            boxes = [int(row["box_number"]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(prefix + "invalid_box") from exc
        if any(box <= 0 for box in boxes) or len(set(boxes)) != len(boxes):
            raise SystemExit(prefix + "duplicate_or_invalid_box")
        field_sizes = [finite(row.get("field_size")) for row in rows]
        if any(size is None or not size.is_integer() or size <= 0 for size in field_sizes):
            raise SystemExit(prefix + "invalid_field_size")
        if len(set(field_sizes)) != 1 or int(field_sizes[0]) != len(rows):
            raise SystemExit(prefix + "incomplete_field")
        probabilities = [finite(row.get("market_implied_probability")) for row in rows]
        if any(probability is None or probability <= 0 for probability in probabilities):
            raise SystemExit(prefix + "invalid_market_probability")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise SystemExit(prefix + "market_probabilities_not_normalized")
        for row in rows:
            capture = parse_source_timestamp(row.get("odds_capture_timestamp"))
            jump = parse_source_timestamp(row.get("jump_at"))
            if capture >= jump:
                raise SystemExit(prefix + "capture_not_before_jump")


def load_history() -> list[dict[str, Any]]:
    sidecar = load_jsonl(HISTORY)
    conn = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        raw_by_id = {int(row[0]): json.loads(row[1] or "{}") for row in conn.execute("SELECT id,raw_row_json FROM csv_dog_history_staging")}
    finally:
        conn.close()
    result = []
    seen = set()
    for row in sidecar:
        native = str(row.get("native_thedogs_dog_id") or "")
        when = str(row.get("source_performance_date") or "")[:10]
        source_id = int(row["source_row_id"])
        raw = raw_by_id.get(source_id)
        if not native or not raw or not when:
            continue
        key = str(row.get("canonical_content_sha256") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        distance = finite(row.get("source_distance_m"))
        elapsed = finite(row.get("individual_time_seconds"))
        sectional = finite(row.get("sectional_1st_seconds"))
        finish = finite(raw.get("PLC"))
        result.append({
            "native_id": native, "date": when, "track": str(row.get("source_track") or ""),
            "distance": int(distance) if distance else None,
            "speed": (distance / elapsed) if distance and elapsed and elapsed > 0 else None,
            "sectional": sectional if sectional and sectional > 0 else None,
            "finish": finish if finish and finish > 0 else None,
            "grade": str(row.get("source_grade") or ""), "source_row_id": source_id,
        })
    return sorted(result, key=lambda r: (r["date"], r["native_id"], r["source_row_id"]))


def build_feature_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    market = [r for r in load_jsonl(MATRIX) if START <= str(r["race_date"]) <= END]
    identities = {exact_key(r): r for r in load_jsonl(TARGET)}
    if len(identities) != len(load_jsonl(TARGET)):
        raise SystemExit("duplicate_target_identity")
    grouped_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in market:
        grouped_market[str(row["race_id"])].append(row)
    validate_canonical_population(grouped_market)
    accepted: list[dict[str, Any]] = []
    excluded = defaultdict(int)
    for race_id, rows in grouped_market.items():
        matches = [identities.get(exact_key(row)) for row in rows]
        if any(match is None for match in matches):
            excluded["incomplete_native_id_field"] += 1
            continue
        native_ids = [str(match["native_thedogs_dog_id"]) for match in matches if match]
        if len(set(native_ids)) != len(native_ids):
            excluded["duplicate_native_id_in_field"] += 1
            continue
        for row, match in zip(rows, matches):
            item = dict(row)
            item["native_thedogs_dog_id"] = str(match["native_thedogs_dog_id"])
            item["target_distance_m"] = int(match["target_distance_m"])
            accepted.append(item)
    history = load_history()
    by_dog: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in history:
        by_dog[row["native_id"]].append(row)
    cache: dict[tuple[str, str, int], tuple[float | None, float | None]] = {}
    def baselines(cutoff: str, track: str, distance: int) -> tuple[float | None, float | None]:
        key = cutoff, track, distance
        if key not in cache:
            eligible = [h for h in history if h["date"] < cutoff and h["track"] == track and h["distance"] == distance]
            cache[key] = (median([h["speed"] for h in eligible if h["speed"]]), median([h["sectional"] for h in eligible if h["sectional"]]))
        return cache[key]
    output = []
    for row in accepted:
        cutoff, track, dist = str(row["race_date"]), str(row["venue"]), int(row["target_distance_m"])
        speed_base, sectional_base = baselines(cutoff, track, dist)
        prior = [h for h in by_dog[row["native_thedogs_dog_id"]] if h["date"] < cutoff]
        prior.sort(key=lambda h: (h["date"], h["source_row_id"]))
        recent = prior[-5:]
        speeds = [h["speed"] / speed_base for h in recent if h["speed"] and speed_base]
        paces = [sectional_base / h["sectional"] for h in recent if h["sectional"] and sectional_base]
        finishes = [h["finish"] for h in recent if h["finish"]]
        same = [h["speed"] / speed_base for h in prior if h["track"] == track and h["distance"] == dist and h["speed"] and speed_base]
        item = {k: row[k] for k in ("race_id", "race_date", "jump_at", "box_number", "field_size", "label_is_winner", "label_finish_position", "market_implied_probability", "canonical_sportsbet_win_odds", "odds_capture_timestamp")}
        item.update({
            "native_thedogs_dog_id": row["native_thedogs_dog_id"],
            "history_cutoff_exclusive": cutoff,
            "speed_median_5": median(speeds), "speed_best_5": max(speeds) if speeds else None,
            "speed_trend_5": slope(speeds), "speed_consistency_5": -float(np.std(speeds)) if len(speeds) >= 2 else None,
            "early_pace_median_5": median(paces), "early_pace_best_5": max(paces) if paces else None,
            "same_track_distance_speed_mean": float(np.mean(same)) if same else None,
            "same_track_distance_starts": len(same),
            "days_since_run": (date.fromisoformat(cutoff) - date.fromisoformat(prior[-1]["date"])).days if prior else None,
            "finish_median_5": median(finishes), "finish_best_5": min(finishes) if finishes else None,
            "box_position": (int(row["box_number"]) - 1) / max(int(row["field_size"]) - 1, 1),
            "normalization_baseline_cutoff_exclusive": cutoff,
        })
        output.append(item)
    by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        by_race[row["race_id"]].append(row)
    for rows in by_race.values():
        available = [r["speed_median_5"] for r in rows if r["speed_median_5"] is not None]
        best = max(available) if available else None
        for row in rows:
            value = row["speed_median_5"]
            row["speed_field_rank"] = (1 + sum(v > value for v in available)) / len(available) if value is not None and available else None
            row["speed_field_gap"] = value - best if value is not None and best is not None else None
    output.sort(key=lambda r: (r["race_date"], r["race_id"], r["box_number"]))
    summary = {
        "source_market_races": len(grouped_market), "accepted_races": len(by_race), "runner_rows": len(output),
        "excluded_races": dict(sorted(excluded.items())),
        "feature_nonmissing": {f: sum(row[f] is not None for row in output) for f in BASE_FEATURES},
        "history_rows": len(history), "date_min": min(r["race_date"] for r in output), "date_max": max(r["race_date"] for r in output),
    }
    return output, summary


def freeze(out: Path) -> None:
    if out.exists():
        raise SystemExit("output_exists")
    out.mkdir(parents=True)
    rows, summary = build_feature_rows()
    matrix_path = out / "feature_matrix.jsonl"
    write_jsonl(matrix_path, rows)
    protocol = {
        "schema_version": "form_speed_market_residual_protocol_v1", "status": "PROTOCOL_FROZEN_READY_TO_EVALUATE",
        "authority": "research_only_no_deployment_betting_promotion", "population": {"start": START, "end": END, **summary},
        "forward_exclusions": {"outcomes_on_or_after": "2026-08-18", "betfair_95_5": "2026-08-18..2026-09-30 untouched", "overround": "eligible no earlier than 2026-10-01 untouched"},
        "identity": "exact race_id+box_number+odds_capture_timestamp to accepted native TheDogs dog ID; complete fields only; no name joins",
        "timing": "every history performance date and every venue-distance normalization observation is strictly before target race_date",
        "features": {"included": list(BASE_FEATURES), "missingness": "training-fold median plus explicit missing indicator", "excluded_after_audit": {"grade_class_change": "cross-jurisdiction grades lack a source-safe common ordinal scale"}},
        "folds": list(FOLDS), "model": {"family": "race-conditional softmax residual", "formula": "softmax(log(Sportsbet normalized p)+X beta)", "l2": L2, "optimizer": "scipy L-BFGS-B", "hyperparameter_search": False},
        "metrics": {"primary": "paired race log-loss delta candidate minus Sportsbet", "secondary": ["multiclass Brier", "ECE 10 equal-width bins", "top-1", "winner rank", "fold stability"], "uncertainty": {"unit": "meeting date", "method": "paired percentile cluster bootstrap", "repetitions": BOOTSTRAPS, "seed": SEED}},
        "predeclared_groups": {"market_rank": ["favourite", "rank_2", "rank_3_plus"], "sportsbet_odds": ["<=3", "3-5", "5-10", ">10"], "non_favourite": "market rank >=2"},
        "economic": {"descriptive_only": True, "stake": "fixed 1 unit", "ev_bins": ["<=0", "0-5%", "5-10%", ">10%"], "one_per_race": "highest strictly positive model-implied EV", "threshold_optimization": False},
        "decision": "FORM_SPEED_RESIDUAL_PROMISING iff combined OOF delta<0, meeting-date bootstrap upper95<0, and at least 2/3 folds improve; else NO_INCREMENTAL_FORM_SPEED_SIGNAL; identity failure yields DATA_IDENTITY_BLOCKED",
        "inputs": {str(p): sha256(p) for p in (MATRIX, TARGET, HISTORY, DB, BETFAIR)},
        "feature_matrix_sha256": sha256(matrix_path), "seed": SEED,
    }
    write_json(out / "protocol.json", protocol)
    write_checksums(out, ["feature_matrix.jsonl", "protocol.json"], "SEALED_SHA256SUMS")


def race_probabilities(rows: Sequence[dict[str, Any]], offsets: np.ndarray) -> np.ndarray:
    result = np.empty(len(rows), dtype=float)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows): groups[row["race_id"]].append(i)
    for idx in groups.values():
        logits = np.log(np.array([rows[i]["market_implied_probability"] for i in idx])) + offsets[idx]
        logits -= logits.max(); p = np.exp(logits); p /= p.sum(); result[idx] = p
    return result


def design(train: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    a, b = [], []
    for f in BASE_FEATURES:
        vals = np.array([np.nan if r[f] is None else float(r[f]) for r in train])
        med = float(np.nanmedian(vals)) if np.isfinite(vals).any() else 0.0
        filled = np.where(np.isfinite(vals), vals, med); sd = float(filled.std()) or 1.0
        for rows, dest in ((train, a), (test, b)):
            raw = np.array([np.nan if r[f] is None else float(r[f]) for r in rows])
            dest.append(np.where(np.isfinite(raw), raw, med)); dest.append((~np.isfinite(raw)).astype(float))
        a[-2] = (a[-2] - med) / sd; b[-2] = (b[-2] - med) / sd
    return np.column_stack(a), np.column_stack(b)


def fit_residual(rows: Sequence[dict[str, Any]], x: np.ndarray) -> np.ndarray:
    y = np.array([r["label_is_winner"] for r in rows], dtype=float)
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        p = race_probabilities(rows, x @ beta)
        value = -float(np.sum(y * np.log(np.clip(p, 1e-15, 1)))) + 0.5 * L2 * float(beta @ beta)
        grad = x.T @ (p - y) + L2 * beta
        return value, grad
    result = minimize(lambda z: objective(z), np.zeros(x.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"optimizer_failed:{result.message}")
    return np.asarray(result.x)


def metrics(rows: Sequence[dict[str, Any]], p: np.ndarray) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows): groups[row["race_id"]].append(i)
    losses=[]; briers=[]; ranks=[]
    y=np.array([r["label_is_winner"] for r in rows],float)
    for idx in groups.values():
        win=next(i for i in idx if y[i] == 1); losses.append(-math.log(max(p[win],1e-15))); briers.append(sum((p[i]-y[i])**2 for i in idx)); ranks.append(1+sum(p[i]>p[win] for i in idx))
    bins=np.minimum((p*10).astype(int),9); ece=sum(np.sum(bins==b)*abs(float(p[bins==b].mean())-float(y[bins==b].mean())) for b in range(10) if np.any(bins==b))/len(p)
    return {"races":len(groups),"runner_rows":len(rows),"log_loss":float(np.mean(losses)),"brier":float(np.mean(briers)),"ece":float(ece),"top_1":float(np.mean(np.array(ranks)==1)),"mean_winner_rank":float(np.mean(ranks))}


def cluster_ci(values: Mapping[str, Sequence[float]], repetitions: int, seed: int) -> tuple[float,float]:
    dates=sorted(values); rng=np.random.default_rng(seed); draws=np.empty(repetitions)
    arrays=[np.asarray(values[d],float) for d in dates]
    for i in range(repetitions):
        sample=rng.integers(0,len(arrays),len(arrays)); draws[i]=np.concatenate([arrays[j] for j in sample]).mean()
    return float(np.percentile(draws,2.5)),float(np.percentile(draws,97.5))


def group_diagnostics(rows: Sequence[dict[str, Any]], candidate: np.ndarray, market: np.ndarray) -> list[dict[str, Any]]:
    ranks={}
    groups:dict[str,list[int]]=defaultdict(list)
    for i,r in enumerate(rows): groups[r["race_id"]].append(i)
    for idx in groups.values():
        for rank,i in enumerate(sorted(idx,key=lambda j:(-market[j],rows[j]["box_number"])),1): ranks[i]=rank
    defs={"favourite":lambda i:ranks[i]==1,"rank_2":lambda i:ranks[i]==2,"rank_3_plus":lambda i:ranks[i]>=3,"non_favourites":lambda i:ranks[i]>=2,"odds_<=3":lambda i:rows[i]["canonical_sportsbet_win_odds"]<=3,"odds_3_5":lambda i:3<rows[i]["canonical_sportsbet_win_odds"]<=5,"odds_5_10":lambda i:5<rows[i]["canonical_sportsbet_win_odds"]<=10,"odds_>10":lambda i:rows[i]["canonical_sportsbet_win_odds"]>10}
    out=[]
    for name,pred in defs.items():
        win_ix=[i for i,r in enumerate(rows) if r["label_is_winner"]==1 and pred(i)]
        out.append({"group":name,"winner_races":len(win_ix),"candidate_minus_market_log_loss":float(np.mean([-math.log(candidate[i])+math.log(market[i]) for i in win_ix])) if win_ix else None})
    return out


def economic(rows: Sequence[dict[str, Any]], p: np.ndarray) -> list[dict[str, Any]]:
    groups:dict[str,list[int]]=defaultdict(list)
    for i,r in enumerate(rows): groups[r["race_id"]].append(i)
    selections=[]
    for idx in groups.values():
        ev=[p[i]*rows[i]["canonical_sportsbet_win_odds"]-1 for i in idx]
        for i,e in zip(idx,ev):
            label="<=0" if e<=0 else "0-5%" if e<=.05 else "5-10%" if e<=.10 else ">10%"
            selections.append((label,i,e,False))
        j=idx[int(np.argmax(ev))]
        if max(ev)>0: selections.append(("highest_positive_one_per_race",j,max(ev),True))
    result=[]
    for label in ("<=0","0-5%","5-10%",">10%","highest_positive_one_per_race"):
        picks=[x for x in selections if x[0]==label]; profits=[rows[i]["canonical_sportsbet_win_odds"]-1 if rows[i]["label_is_winner"] else -1 for _,i,_,_ in picks]
        by_date:dict[str,list[float]]=defaultdict(list)
        for (_,i,_,_),profit in zip(picks,profits): by_date[rows[i]["race_date"]].append(profit)
        ci=cluster_ci(by_date,ROI_BOOTSTRAPS,SEED+91) if by_date else (None,None)
        cumulative=np.cumsum(profits) if profits else np.array([]); peak=np.maximum.accumulate(np.r_[0,cumulative]); drawdown=float(np.max(peak[1:]-cumulative)) if profits else 0.0
        result.append({"bin":label,"count":len(picks),"wins":sum(rows[i]["label_is_winner"] for _,i,_,_ in picks),"win_rate":float(np.mean([rows[i]["label_is_winner"] for _,i,_,_ in picks])) if picks else None,"mean_odds":float(np.mean([rows[i]["canonical_sportsbet_win_odds"] for _,i,_,_ in picks])) if picks else None,"pnl":float(sum(profits)),"roi":float(np.mean(profits)) if profits else None,"roi_ci95":list(ci),"max_drawdown_units":drawdown})
    return result


def complete_betfair_overlap(
    predictions: Sequence[dict[str, Any]], betfair_rows: Sequence[dict[str, Any]]
) -> tuple[list[int], dict[tuple[str, int], dict[str, Any]], int]:
    betfair: dict[tuple[str, int], dict[str, Any]] = {}
    for row in betfair_rows:
        key = str(row["race_id"]), int(row["box_number"])
        if key in betfair:
            raise SystemExit("duplicate_betfair_overlap_identity")
        betfair[key] = row
    prediction_groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(predictions):
        prediction_groups[str(row["race_id"])].append(i)
    betfair_boxes: dict[str, set[int]] = defaultdict(set)
    for race_id, box_number in betfair:
        betfair_boxes[race_id].add(box_number)
    retained: list[int] = []
    partial_races = 0
    for race_id, idx in prediction_groups.items():
        keys = [(str(predictions[i]["race_id"]), int(predictions[i]["box_number"])) for i in idx]
        prediction_boxes = {key[1] for key in keys}
        prices = [finite(betfair[key].get("betfair_scheduled_off_back_price")) for key in keys if key in betfair]
        if (
            prediction_boxes == betfair_boxes.get(race_id, set())
            and len(prices) == len(keys)
            and all(price is not None and price > 0 for price in prices)
        ):
            retained.extend(idx)
        elif prediction_boxes & betfair_boxes.get(race_id, set()):
            partial_races += 1
    return retained, betfair, partial_races


def repo_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {
        "base_commit": "724c5cc3cba45289226b14440a871d63af3e6db0",
        "head_commit": git("rev-parse", "HEAD"),
        "head_tree": git("rev-parse", "HEAD^{tree}"),
        "index_tree": git("write-tree"),
        "dirty": bool(git("status", "--porcelain=v1", "--untracked-files=all")),
    }


def evaluate(out: Path) -> None:
    protocol=json.loads((out/"protocol.json").read_text()); matrix_path=out/"feature_matrix.jsonl"
    if sha256(matrix_path)!=protocol["feature_matrix_sha256"]: raise SystemExit("sealed_feature_matrix_hash_mismatch")
    rows=load_jsonl(matrix_path); predictions=[]; fold_reports=[]
    for fold in FOLDS:
        train=[r for r in rows if fold["train_start"]<=r["race_date"]<=fold["train_end"]]
        test=[r for r in rows if fold["test_start"]<=r["race_date"]<=fold["test_end"]]
        xtrain,xtest=design(train,test); beta=fit_residual(train,xtrain)
        candidate=race_probabilities(test,xtest@beta); market=np.array([r["market_implied_probability"] for r in test])
        fold_reports.append({**fold,"train_races":len({r["race_id"] for r in train}),"test_races":len({r["race_id"] for r in test}),"candidate":metrics(test,candidate),"sportsbet":metrics(test,market),"log_loss_delta":metrics(test,candidate)["log_loss"]-metrics(test,market)["log_loss"]})
        for row,pc,pm in zip(test,candidate,market): predictions.append({**row,"fold_id":fold["id"],"candidate_probability":float(pc),"sportsbet_probability":float(pm),"candidate_implied_ev":float(pc*row["canonical_sportsbet_win_odds"]-1)})
    if len({(r["race_id"],r["box_number"]) for r in predictions})!=len(predictions): raise SystemExit("duplicate_oof_prediction")
    cand=np.array([r["candidate_probability"] for r in predictions]); market=np.array([r["sportsbet_probability"] for r in predictions])
    deltas:dict[str,list[float]]=defaultdict(list)
    for i,r in enumerate(predictions):
        if r["label_is_winner"]: deltas[r["race_date"]].append(-math.log(cand[i])+math.log(market[i]))
    delta=float(np.mean([v for vals in deltas.values() for v in vals])); ci=cluster_ci(deltas,BOOTSTRAPS,SEED)
    improved=sum(f["log_loss_delta"]<0 for f in fold_reports); decision="FORM_SPEED_RESIDUAL_PROMISING" if delta<0 and ci[1]<0 and improved>=2 else "NO_INCREMENTAL_FORM_SPEED_SIGNAL"
    betfair_rows=load_jsonl(BETFAIR); overlap,bf,partial_betfair_races=complete_betfair_overlap(predictions,betfair_rows)
    consensus=np.zeros(len(overlap)); overlap_rows=[predictions[i] for i in overlap]
    by_race:dict[str,list[int]]=defaultdict(list)
    for j,r in enumerate(overlap_rows): by_race[r["race_id"]].append(j)
    for idx in by_race.values():
        raw=np.array([1/bf[(overlap_rows[j]["race_id"],overlap_rows[j]["box_number"])]["betfair_scheduled_off_back_price"] for j in idx]); raw/=raw.sum()
        consensus[idx]=.95*np.array([overlap_rows[j]["sportsbet_probability"] for j in idx])+.05*raw
    report={"schema_version":"form_speed_market_residual_report_v1","decision":decision,"repo":repo_metadata(),"protocol_sha256":sha256(out/"protocol.json"),"population":protocol["population"],"folds":fold_reports,"combined_oof":{"candidate":metrics(predictions,cand),"sportsbet":metrics(predictions,market),"log_loss_delta":delta,"meeting_date_cluster_bootstrap_ci95":list(ci),"folds_improved":improved},"predeclared_groups":group_diagnostics(predictions,cand,market),"economic_diagnostic":economic(predictions,cand),"betfair_95_5_diagnostic":{"retuned":False,"selected_on":False,"races":len(by_race),"partial_races_excluded":partial_betfair_races,"candidate":metrics(overlap_rows,cand[overlap]),"sportsbet":metrics(overlap_rows,market[overlap]),"frozen_95_5":metrics(overlap_rows,consensus)},"boundaries":{"outcomes_2026_08_18_or_later_opened":False,"forward_cohorts_touched":False,"deployment":False,"betting":False,"promotion":False},"findings":{"BLOCKING":[],"IMPORTANT":["Development OOF only; the same historical corpus has supported earlier exploratory branches.","Grade/class change was excluded before fitting because source grades are not safely ordinal across jurisdictions."],"OPTIONAL":["The strongest next experiment is a genuinely prospective, source-receipted form/speed residual test only if this frozen gate passes; otherwise acquire a novel independent pre-jump source rather than retuning these features."]},"supported_claims":["Exact native-ID prior history and as-of target-date normalization were evaluated only as a residual correction to corrected Sportsbet WIN."],"unsupported_claims":["No forward, betting, ROI-selection, deployment, promotion, or causal claim."]}
    write_jsonl(out/"oof_predictions.jsonl",predictions); write_json(out/"report.json",report); write_checksums(out,["feature_matrix.jsonl","protocol.json","oof_predictions.jsonl","report.json"],"SHA256SUMS")


def write_checksums(out: Path, names: Sequence[str], manifest: str) -> None:
    (out/manifest).write_text("".join(f"{sha256(out/name)}  {name}\n" for name in names),encoding="utf-8")


def verify(out: Path) -> None:
    for manifest in ("SEALED_SHA256SUMS","SHA256SUMS"):
        with (out/manifest).open() as f:
            for line in f:
                digest,name=line.rstrip().split("  ",1)
                if sha256(out/name)!=digest: raise SystemExit(f"checksum_mismatch:{name}")
    report=json.loads((out/"report.json").read_text()); rows=load_jsonl(out/"oof_predictions.jsonl")
    if any(r["race_date"]>END for r in rows): raise SystemExit("forward_outcome_boundary_breached")
    if len({(r["race_id"],r["box_number"]) for r in rows})!=len(rows): raise SystemExit("oof_not_unique")
    if report["decision"] not in {"FORM_SPEED_RESIDUAL_PROMISING","NO_INCREMENTAL_FORM_SPEED_SIGNAL","DATA_IDENTITY_BLOCKED"}: raise SystemExit("invalid_decision")


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=DEFAULT_OUT); group=ap.add_mutually_exclusive_group(required=True); group.add_argument("--freeze",action="store_true"); group.add_argument("--evaluate",action="store_true"); group.add_argument("--verify",action="store_true"); args=ap.parse_args()
    if args.freeze: freeze(args.output)
    elif args.evaluate: evaluate(args.output)
    else: verify(args.output)


if __name__ == "__main__": main()
