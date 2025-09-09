#!/usr/bin/env python3
"""
Backtest value thresholds for EV and edge using real predictions and live odds.

- Joins predictions and current/pre-off live odds by (race_id, dog_clean_name)
- Uses race_metadata to filter odds to timestamps before race off (if available)
- Computes implied probs (overround-adjusted), EV, edge, and simple unit-stake ROI
- Sweeps EV and edge thresholds to find robust candidates

Usage:
    python scripts/backtest_value_thresholds.py --hours 48 --ev 0.0,0.02,0.05,0.1 --edge 0.0,0.01,0.02,0.05

Notes:
- Does NOT use odds to train models; strictly post-processing for policy tuning
- Ensure that predictions and live_odds tables exist; otherwise exits gracefully
"""
import argparse
import sqlite3
from datetime import datetime, timedelta
import math

import os

DB_PATH = os.getenv("DATABASE_PATH", "greyhound_racing_data.db")


def _pick_db_path(default_path: str) -> str:
    candidates = [
        os.getenv("DATABASE_PATH"),
        os.path.join(os.getcwd(), "greyhound_racing_data_writable.db"),
        os.path.join(os.getcwd(), default_path),
    ]
    for p in candidates:
        if not p:
            continue
        try:
            if os.path.exists(p):
                # quick sanity: has live_odds table?
                conn = sqlite3.connect(p)
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='live_odds'")
                    if cur.fetchone():
                        return p
                finally:
                    conn.close()
        except Exception:
            continue
    # fallback to default
    return default_path


def _q(conn: sqlite3.Connection, sql: str, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]


def _norm_name(s: str) -> str:
    import re
    return re.sub(r"[^\w\s]", "", (s or "").upper().strip())


def implied_probs_from_odds(recs):
    # Compute implied and normalized per race
    by_race = {}
    for r in recs:
        by_race.setdefault(r["race_id"], []).append(r)
    for race_id, rows in by_race.items():
        raw = []
        for r in rows:
            od = r.get("odds_decimal")
            r["implied_prob_raw"] = (1.0 / float(od)) if od and od > 0 else 0.0
            raw.append(r["implied_prob_raw"])
        s = sum(raw)
        for r in rows:
            r["implied_prob_norm"] = (r["implied_prob_raw"]/s) if s > 1e-12 else 0.0
    return recs


def renorm_model_probs(preds):
    by_race = {}
    for p in preds:
        by_race.setdefault(p["race_id"], []).append(p)
    for race_id, rows in by_race.items():
        vals = []
        for r in rows:
            pm = None
            for k in ("win_prob_norm", "win_probability", "win_prob", "final_score", "predicted_probability", "prediction_score"):
                v = r.get(k)
                if v is None:
                    continue
                try:
                    x = float(v)
                    pm = x/100.0 if x > 1.0 else x
                    break
                except Exception:
                    continue
            if pm is None:
                pm = 0.0
            r["p_model_raw"] = max(0.0, min(1.0, pm))
            vals.append(r["p_model_raw"])
        s = sum(vals)
        for r in rows:
            r["p_model_norm"] = (r["p_model_raw"]/s) if s > 1e-12 else (1.0/max(1,len(rows)))
    return preds


def logit_blend(model_p, market_p, alpha):
    def _safe_logit(p: float) -> float:
        p = min(1-1e-9, max(1e-9, p))
        return math.log(p/(1-p))
    def _sigmoid(z: float) -> float:
        return 1.0/(1.0+math.exp(-z))
    z = alpha*_safe_logit(model_p) + (1.0-alpha)*_safe_logit(market_p if market_p>0 else 0.0001)
    return _sigmoid(z)


def backtest(hours, ev_grid, edge_grid, alpha=None):
    dbp = _pick_db_path(DB_PATH)
    conn = sqlite3.connect(dbp)
    print(f"Using database: {dbp}")
    conn.row_factory = sqlite3.Row
    now = datetime.now()
    since = now - timedelta(hours=hours)
    try:
        # Pull predictions (recent)
        preds = _q(conn, """
            SELECT race_id, dog_clean_name, predicted_probability AS predicted_probability,
                   confidence_level, timestamp
            FROM predictions
            WHERE timestamp >= ?
        """, (since.strftime("%Y-%m-%d %H:%M:%S"),))
        # Fallback: read from ./predictions JSON files
        if not preds:
            import os, json, time as _time
            pred_dir = os.path.join(os.getcwd(), "predictions")
            if os.path.isdir(pred_dir):
                files = []
                now_ts = _time.time()
                for fn in os.listdir(pred_dir):
                    if fn.endswith(".json"):
                        fp = os.path.join(pred_dir, fn)
                        try:
                            mtime = os.path.getmtime(fp)
                            if mtime >= now_ts - hours*3600:
                                files.append((fp, mtime))
                        except Exception:
                            continue
                files.sort(key=lambda x: x[1], reverse=True)
                preds = []
                for fp, _ in files[:100]:
                    try:
                        with open(fp, "r") as f:
                            data = json.load(f)
                        race_id = (
                            data.get("race_context", {}).get("race_id")
                            or data.get("race_id")
                            or None
                        )
                        ts = data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for d in data.get("predictions", []):
                            name = d.get("dog_clean_name") or d.get("dog_name")
                            if not race_id or not name:
                                continue
                            preds.append({
                                "race_id": race_id,
                                "dog_clean_name": name,
                                "predicted_probability": d.get("win_prob_norm") or d.get("win_probability") or d.get("final_score") or d.get("prediction_score"),
                                "confidence_level": d.get("confidence_label") or d.get("confidence_level") or "MEDIUM",
                                "timestamp": ts,
                            })
                    except Exception:
                        continue
        if not preds:
            print("No recent predictions found (DB and files)")
            return
        # Pull live odds snapshots and restrict to last odds before race off if possible
        odds = _q(conn, """
            SELECT lo.race_id, lo.dog_clean_name, lo.odds_decimal, lo.timestamp,
                   rm.race_date, rm.race_time
            FROM live_odds lo
            LEFT JOIN race_metadata rm ON lo.race_id = rm.race_id
            WHERE lo.timestamp >= ?
        """, (since.strftime("%Y-%m-%d %H:%M:%S"),))
        if not odds:
            print("No odds records found")
            return
        # Build a pre-off map: keep the latest odds before race_time
        by_key = {}
        for o in odds:
            rid = o["race_id"]
            dog = _norm_name(o["dog_clean_name"])
            ts = o.get("timestamp")
            race_dt = None
            try:
                if o.get("race_date") and o.get("race_time"):
                    race_dt = datetime.strptime(f"{o['race_date']} {o['race_time']}", "%Y-%m-%d %H:%M")
            except Exception:
                race_dt = None
            if race_dt is not None and ts is not None:
                try:
                    tsdt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
                if tsdt <= race_dt:
                    key = (rid, dog)
                    if key not in by_key or by_key[key]["timestamp"] < ts:
                        by_key[key] = dict(o)
        if not by_key:
            # Fallback: use latest odds per (race_id, dog_clean_name) regardless of race_time
            print("No pre-off odds after filtering; falling back to latest odds per runner")
            latest = {}
            for o in odds:
                key = (o["race_id"], _norm_name(o["dog_clean_name"]))
                ts = o.get("timestamp") or ""
                if key not in latest or (ts and latest[key].get("timestamp","") < ts):
                    latest[key] = dict(o)
            odds_list = list(latest.values())
        else:
            odds_list = list(by_key.values())
        implied_probs_from_odds(odds_list)
        # Join preds and odds
        pred_map = {(p["race_id"], _norm_name(p["dog_clean_name"])): p for p in preds}
        joined = []
        for o in odds_list:
            key = (o["race_id"], _norm_name(o["dog_clean_name"]))
            if key in pred_map:
                r = {**pred_map[key], **o}
                joined.append(r)
        if not joined:
            print("No joined prediction/odds records")
            return
        renorm_model_probs(joined)
        # Compute blended prob
        for r in joined:
            pm = r["p_model_norm"]
            mk = r["implied_prob_norm"]
            if alpha is not None and 0.0 <= alpha <= 1.0:
                pb = logit_blend(pm, mk, alpha)
            else:
                pb = pm
            r["p_blend"] = pb
            r["edge"] = pb - mk
            od = r.get("odds_decimal") or 0.0
            r["ev_win"] = pb*od - 1.0 if od>0 else None
        # ROI calc: simple unit stakes, payoff = odds if win else 0 (this assumes availability of results table)
        # If results table exists, fetch winners; else skip ROI calc and just output threshold hit counts.
        winners = _q(conn, """
            SELECT race_id, UPPER(REPLACE(REPLACE(dog_name,'-',' '),'_',' ')) AS dog_clean_name
            FROM race_results
            WHERE timestamp >= ?
        """, (since.strftime("%Y-%m-%d %H:%M:%S"),))
        winset = set((w["race_id"], _norm_name(w["dog_clean_name"])) for w in winners)

        def eval_thresholds(ev_t, edge_t):
            picks = [r for r in joined if (r.get("ev_win") is not None and r["ev_win"]>=ev_t) or (r.get("edge") is not None and r["edge"]>=edge_t)]
            if not picks:
                return {"bets":0,"roi":None}
            if not winset:
                return {"bets":len(picks),"roi":None}
            stake = 0.0
            ret = 0.0
            for p in picks:
                stake += 1.0
                if (p["race_id"], _norm_name(p["dog_clean_name"])) in winset:
                    ret += float(p.get("odds_decimal") or 0.0)
            roi = (ret - stake) / stake if stake>0 else None
            return {"bets":len(picks),"roi":roi}

        print("Threshold sweep (ev, edge) -> bets, roi:")
        for ev_t in ev_grid:
            for ed_t in edge_grid:
                res = eval_thresholds(ev_t, ed_t)
                print(f"ev>={ev_t:.3f} edge>={ed_t:.3f} -> bets={res['bets']} roi={res['roi']}")
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--ev", type=str, default="0.0,0.02,0.05,0.10")
    ap.add_argument("--edge", type=str, default="0.0,0.01,0.02,0.05")
    ap.add_argument("--alpha", type=float, default=None, help="logit-blend alpha in [0,1]")
    args = ap.parse_args()
    ev_grid = [float(x) for x in args.ev.split(",") if x.strip()]
    edge_grid = [float(x) for x in args.edge.split(",") if x.strip()]
    backtest(args.hours, ev_grid, edge_grid, alpha=args.alpha)


if __name__ == "__main__":
    main()

