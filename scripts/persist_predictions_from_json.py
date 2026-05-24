#!/usr/bin/env python3
"""
Persist predictions from JSON files under ./predictions into the SQLite DB predictions table.
- Attempts to map JSON race info to the standardized race_id by consulting race_metadata
- Normalizes dog names to DOG_CLEAN_NAME (uppercase, alnum+space)
- Chooses predicted_probability from win_prob_norm/win_probability/final_score/prediction_score

Usage:
  python scripts/persist_predictions_from_json.py --hours 72

Environment:
  DATABASE_PATH (optional) to override DB; defaults to auto-picking writable DB if present.
"""
import argparse
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta


def _norm_name(s: str) -> str:
  return re.sub(r"[^\w\s]", "", (s or "").upper().strip())


VENUE_CODE_MAP = {
  # Common AU greyhound codes to canonical venue names (expand as needed)
  "WRGL": "WARRAGUL",
  "GOSF": "GOSFORD",
  "AP_K": "ANGLE PARK",
  "AP": "ANGLE PARK",
  "GEE": "GEELONG",
  "BAL": "BALLARAT",
  "BEN": "BENDIGO",
  "HEA": "HEALESVILLE",
  "MEA": "THE MEADOWS",
  "SAN": "SANDOWN PARK",
  "HOR": "HORSHAM",
  "RICH": "RICHMOND",
  "RICH_STRAIGHT": "RICHMOND STRAIGHT",
}

def _canon_venue(code_or_name: str) -> str:
  s = (code_or_name or "").strip().upper().replace("-", " ")
  if s in VENUE_CODE_MAP:
    return VENUE_CODE_MAP[s]
  return s

def _pick_db_path(default_candidates):
  for p in default_candidates:
    if not p:
      continue
    try:
      if os.path.exists(p):
        return p
    except Exception:
      continue
  return default_candidates[-1]


def _ensure_table(conn: sqlite3.Connection):
  conn.execute(
    """
    CREATE TABLE IF NOT EXISTS predictions (
      race_id TEXT,
      dog_clean_name TEXT,
      predicted_probability REAL,
      confidence_level TEXT,
      timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """
  )
  conn.commit()


def _load_rm_index(conn: sqlite3.Connection):
  rm = []
  try:
    cur = conn.cursor()
    cur.execute("SELECT race_id, venue, race_date, race_number FROM race_metadata")
    cols = [d[0] for d in cur.description]
    for r in cur.fetchall():
      rm.append(dict(zip(cols, r)))
  except Exception:
    pass
  by_meta = {}
  for r in rm:
    key = (_norm_name(r.get("venue") or ""), str(r.get("race_date") or ""), int(r.get("race_number") or 0))
    by_meta[key] = r.get("race_id")
  return by_meta


def _ensure_ledger_table(conn: sqlite3.Connection):
  try:
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS processed_prediction_files (
        path TEXT PRIMARY KEY,
        mtime REAL,
        size INTEGER,
        processed_at TEXT DEFAULT CURRENT_TIMESTAMP
      )
      """
    )
  except Exception:
    pass


def _ledger_get(conn: sqlite3.Connection, path: str):
  try:
    cur = conn.cursor()
    cur.execute("SELECT mtime, size FROM processed_prediction_files WHERE path=?", (path,))
    return cur.fetchone()
  except Exception:
    return None


def _ledger_upsert(conn: sqlite3.Connection, path: str, mtime: float, size: int):
  try:
    conn.execute(
      """
      INSERT INTO processed_prediction_files (path, mtime, size)
      VALUES (?, ?, ?)
      ON CONFLICT(path) DO UPDATE SET
        mtime=excluded.mtime,
        size=excluded.size,
        processed_at=CURRENT_TIMESTAMP
      """,
      (path, float(mtime), int(size)),
    )
  except Exception:
    pass


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--hours", type=int, default=72)
  ap.add_argument("--no-ledger", action="store_true", help="Disable skip-ledger; always process files")
  args = ap.parse_args()

  # DB selection
  db_env = os.getenv("DATABASE_PATH")
  candidates = [
    db_env,
    os.path.join(os.getcwd(), "greyhound_racing_data_writable.db"),
    os.path.join(os.getcwd(), "greyhound_racing_data.db"),
  ]
  db_path = _pick_db_path(candidates)
  conn = sqlite3.connect(db_path)
  print(f"Using database: {db_path}")
  try:
    _ensure_table(conn)
    _ensure_ledger_table(conn)
    by_meta = _load_rm_index(conn)

    pred_dir = os.path.join(os.getcwd(), "predictions")
    if not os.path.isdir(pred_dir):
      print("No predictions directory found")
      return

    now_ts = datetime.now().timestamp()
    files = []
    for fn in os.listdir(pred_dir):
      if fn.endswith(".json") and "summary" not in fn:
        fp = os.path.join(pred_dir, fn)
        try:
          mtime = os.path.getmtime(fp)
          if mtime >= now_ts - args.hours * 3600:
            files.append((fp, mtime))
        except Exception:
          continue
    files.sort(key=lambda x: x[1], reverse=True)

    use_ledger = not args.no_ledger
    inserted = 0
    for fp, mtime in files:
      try:
        size = os.path.getsize(fp)
      except Exception:
        size = -1

      if use_ledger:
        row = _ledger_get(conn, fp)
        if row is not None:
          prev_mtime, prev_size = row
          if prev_mtime == mtime and prev_size == size:
            print(f"Skip (already processed): {fp}")
            continue

      try:
        with open(fp, "r") as f:
          data = json.load(f)
      except Exception as e:
        print(f"Skip {fp}: {e}")
        continue

      # Race metadata from JSON
      race_id = data.get("race_id")
      ri = data.get("race_info") or {}
      rc = data.get("race_context") or {}
      venue_code = _canon_venue(ri.get("venue") or rc.get("venue") or "")
      race_number = ri.get("race_number") or rc.get("race_number")
      race_date = ri.get("race_date") or rc.get("race_date")
      # Parse filename if needed: 'Race N - CODE - YYYY-MM-DD.csv'
      fn = ri.get("filename") or rc.get("filename") or data.get("race_id") or ""
      base = str(fn).replace(".csv", "")
      m = re.match(r".*?Race\s+(\d+)\s+-\s+([A-Z0-9_\-]+)\s+-\s+(\d{4}-\d{2}-\d{2}).*", base)
      if m:
        race_number = race_number or m.group(1)
        venue_code = venue_code or _canon_venue(m.group(2))
        race_date = race_date or m.group(3)
      try:
        race_number = int(race_number) if race_number is not None else None
      except Exception:
        race_number = None

      # Map to standardized race_id using DB race_metadata when possible
      std_race_id = race_id
      key = (_norm_name(venue_code), str(race_date), int(race_number or 0))
      if venue_code and race_date and race_number and key in by_meta:
        std_race_id = by_meta[key]

      preds = data.get("predictions") or []
      if not isinstance(preds, list):
        # Update ledger even if file didn't contain predictions, to avoid re-reading unchanged files
        if use_ledger:
          _ledger_upsert(conn, fp, mtime, size)
          conn.commit()
        continue
      cur = conn.cursor()
      for p in preds:
        nm = _norm_name(p.get("dog_clean_name") or p.get("dog_name") or p.get("name"))
        if not nm:
          continue
        # choose predicted_probability
        prob = None
        for k in ("win_prob_norm", "win_probability", "final_score", "prediction_score"):
          v = p.get(k)
          if v is None:
            continue
          try:
            x = float(v)
            prob = x / 100.0 if x > 1.0 else x
            break
          except Exception:
            continue
        if prob is None:
          continue
        conf = p.get("confidence_label") or p.get("confidence_level") or "MEDIUM"
        try:
          cur.execute(
            """
            INSERT INTO predictions (race_id, dog_clean_name, predicted_probability, confidence_level, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (std_race_id or race_id or base, nm, float(prob), str(conf), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
          )
          inserted += 1
        except Exception as e:
          print(f"Insert failed for {std_race_id}:{nm}: {e}")
      conn.commit()

      if use_ledger:
        _ledger_upsert(conn, fp, mtime, size)
        conn.commit()

    print(f"Inserted {inserted} prediction rows from JSON files")
  finally:
    conn.close()


if __name__ == "__main__":
  main()

