"""
Prediction Pipeline V4 - Advanced Integrated System
==================================================

An advanced prediction pipeline based on ML System V4, leveraging all available model improvements
and EV calculations for enhanced predictions.
"""

import logging
import os
import threading
from datetime import datetime

import pandas as pd
import sqlite3
from typing import Any, Optional

from ml_system_v4 import MLSystemV4
from utils.feature_flags import load_flags
from utils.leakage_guard import strip_target_leakage_columns
from src.parsers.csv_ingestion import CsvIngestion

# --- Helpers for participant detection and normalization ---
import re as _re

# Accept optional whitespace, 1-2 digits, optional '.', ')', ':', or '-' then optional whitespace
_NUM_PREFIX_RE = _re.compile(r"^\s*(\d{1,2})\s*[\.\):-]\s*")

def _has_numeric_prefix(name: str) -> bool:
    try:
        return bool(_NUM_PREFIX_RE.match(str(name) or ""))
    except Exception:
        return False

def _normalize_dog_name_no_prefix(name: str) -> str:
    """Normalize dog name for grouping, removing numeric header prefix and punctuation.
    Returns Title Case string with collapsed whitespace.
    """
    try:
        s = str(name or "")
        # Strip various quotes/unicode punctuation and normalize spaces
        for a, b in [
            ("\u201c", ""), ("\u201d", ""), ("\u2018", ""), ("\u2019", ""),
            ("\u2013", "-"), ("\u2014", "-"), ('"', ''), ("'", ''), ("`", ""),
            ("\u00A0", " ")  # non-breaking space to normal space
        ]:
            s = s.replace(a, b)
        s = s.strip()
        # Remove numeric prefix like "2. ", "3) ", "4- ", "5: "
        m = _NUM_PREFIX_RE.match(s)
        if m:
            s = s[m.end():]
        # Collapse internal whitespace
        s = _re.sub(r"\s+", " ", s).strip()
        # Title-case to match DB format
        return s.title()
    except Exception:
        return str(name or "").strip()


def _normalize_odds_name_key(name: str) -> str:
    """Normalize dog names for market-odds joins across punctuation/case variants."""
    try:
        return _re.sub(r"[^A-Z0-9]", "", str(name or "").upper())
    except Exception:
        return str(name or "").upper().replace(" ", "")


def _store_market_odds(odds_map: dict[str, float], dog_name, odds) -> None:
    """Store odds under exact, uppercase, and punctuation-free dog-name aliases."""
    try:
        odds_value = float(odds)
    except Exception:
        return

    base = str(dog_name or "").strip()
    keys = {base, base.upper(), _normalize_odds_name_key(base)}
    for key in keys:
        if key:
            odds_map[key] = odds_value


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        parsed = float(value)
        if pd.isna(parsed):
            return default
        return parsed
    except Exception:
        return default


def _prediction_name_aliases(prediction: dict[str, Any]) -> tuple[str, ...]:
    dog_name = (
        prediction.get("dog_clean_name")
        or prediction.get("dog_name")
        or prediction.get("name")
        or ""
    )
    dog_text = str(dog_name).strip()
    return (dog_text, dog_text.upper(), _normalize_odds_name_key(dog_text))


def _lookup_market_odds(odds_map: dict[str, float], prediction: dict[str, Any]) -> float | None:
    for key in _prediction_name_aliases(prediction):
        if key in odds_map:
            return _safe_float(odds_map[key])
    return None


def _append_quality_flag(prediction: dict[str, Any], flag: str) -> None:
    flags = prediction.get("quality_flags")
    if not isinstance(flags, list):
        flags = []
    if flag not in flags:
        flags.append(flag)
    prediction["quality_flags"] = flags


def _market_disagreement_threshold() -> float:
    try:
        return max(0.0, float(os.getenv("V4_MARKET_DISAGREEMENT_DELTA", "0.08")))
    except Exception:
        return 0.08


def _annotate_market_context(
    predictions: list[dict[str, Any]], win_odds: dict[str, float]
) -> dict[str, Any]:
    """Attach market-implied probabilities and disagreement flags without reranking."""
    if not predictions or not win_odds:
        return {
            "market_odds_count": 0,
            "market_implied_overround": None,
            "large_disagreement_count": 0,
            "large_disagreement_threshold": _market_disagreement_threshold(),
        }

    implied_by_index: dict[int, float] = {}
    odds_count = 0
    for idx, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            continue
        odds_value = _lookup_market_odds(win_odds, prediction)
        if odds_value is None or odds_value <= 0:
            continue
        odds_count += 1
        implied = 1.0 / float(odds_value)
        implied_by_index[idx] = implied
        prediction["market_odds_win"] = float(odds_value)
        prediction.setdefault("odds_win", float(odds_value))
        prediction["odds_implied_prob"] = float(implied)

    implied_total = sum(implied_by_index.values())
    threshold = _market_disagreement_threshold()
    large_count = 0

    for idx, implied in implied_by_index.items():
        prediction = predictions[idx]
        implied_norm = implied / implied_total if implied_total > 0 else None
        prediction["odds_implied_prob_norm"] = (
            float(implied_norm) if implied_norm is not None else None
        )

        model_prob = _safe_float(
            prediction.get("win_prob_norm", prediction.get("win_probability"))
        )
        if model_prob is None or implied_norm is None:
            continue

        delta = float(model_prob) - float(implied_norm)
        prediction["model_market_prob_delta"] = delta
        prediction["model_market_prob_delta_abs"] = abs(delta)
        if abs(delta) >= threshold:
            large_count += 1
            _append_quality_flag(prediction, "large_model_market_disagreement")
            prediction["market_disagreement_warning"] = (
                "model probability differs materially from normalized market probability"
            )

    return {
        "market_odds_count": odds_count,
        "market_implied_overround": float(implied_total) if odds_count else None,
        "large_disagreement_count": large_count,
        "large_disagreement_threshold": threshold,
    }


def _extract_box_from_name_or_row(raw_name: str, row: dict) -> Optional[int]:
    """Get box number from numeric prefix or BOX column.
    Returns None if not determinable.
    """
    try:
        m = _NUM_PREFIX_RE.match(str(raw_name or ""))
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
        # Fallback to BOX-like columns
        for key in ("BOX", "Box", "box", "box_number"):
            if key in row and row.get(key) not in (None, ""):
                try:
                    val = int(pd.to_numeric(row.get(key), errors="coerce"))
                    return val if pd.notna(val) else None
                except Exception:
                    continue
    except Exception:
        return None
    return None

logger = logging.getLogger(__name__)

_ML_SYSTEM_V4_CACHE_LOCK = threading.Lock()
_ML_SYSTEM_V4_CACHE = {}
_EMBEDDED_HISTORY_INGEST_LOCK = threading.Lock()
_EMBEDDED_HISTORY_INGESTED_KEYS = set()


def _truthy_env(name: str, default: str = "0") -> bool:
    try:
        return str(os.getenv(name, default)).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    except Exception:
        return False


def _resolved_path_for_cache(path: str) -> str:
    try:
        return os.path.realpath(os.path.abspath(path))
    except Exception:
        return str(path)


def _ml_system_cache_key(db_path: str) -> tuple:
    return (
        _resolved_path_for_cache(db_path),
        os.getenv("V4_MODEL_PATH") or "",
        os.getenv("PINNED_MODEL_ID") or "",
        os.getenv("V4_DISABLE_ACCURACY_OPTIMIZER") or "",
        os.getenv("TGR_ENABLED") or "",
        os.getenv("PREDICTION_IMPORT_MODE") or "",
        os.getenv("ENABLE_RESULTS_SCRAPERS") or "",
        os.getenv("GREYHOUND_LOOKBACK_DAYS") or "",
    )


def _get_cached_ml_system_v4(db_path: str) -> MLSystemV4:
    if _truthy_env("V4_DISABLE_ML_SYSTEM_CACHE"):
        return MLSystemV4(db_path)

    key = _ml_system_cache_key(db_path)
    with _ML_SYSTEM_V4_CACHE_LOCK:
        cached = _ML_SYSTEM_V4_CACHE.get(key)
        if cached is not None:
            logger.info("♻️ Reusing cached MLSystemV4 for %s", key[0])
            return cached
        system = MLSystemV4(db_path)
        _ML_SYSTEM_V4_CACHE[key] = system
        return system


def register_cached_ml_system_v4(db_path: str, system: MLSystemV4) -> None:
    """Seed the shared V4 cache with an ML system loaded by another owner."""
    if system is None or _truthy_env("V4_DISABLE_ML_SYSTEM_CACHE"):
        return
    key = _ml_system_cache_key(db_path)
    with _ML_SYSTEM_V4_CACHE_LOCK:
        _ML_SYSTEM_V4_CACHE.setdefault(key, system)


def get_cached_ml_system_status() -> dict:
    """Lightweight process-local model load status for health endpoints."""
    with _ML_SYSTEM_V4_CACHE_LOCK:
        items = list(_ML_SYSTEM_V4_CACHE.items())

    systems = []
    for key, system in items:
        try:
            info = getattr(system, "model_info", {}) or {}
            systems.append(
                {
                    "db_path": key[0],
                    "loaded": bool(getattr(system, "calibrated_pipeline", None)),
                    "model_id": info.get("model_id"),
                    "model_version": info.get("model_version"),
                    "source": info.get("source"),
                    "feature_count": len(getattr(system, "feature_columns", []) or []),
                }
            )
        except Exception:
            systems.append({"db_path": key[0], "loaded": False})

    return {
        "cache_enabled": not _truthy_env("V4_DISABLE_ML_SYSTEM_CACHE"),
        "cached_systems": len(systems),
        "systems": systems,
    }


def _embedded_history_ingest_key(db_path: str, race_file_path: str) -> tuple:
    try:
        st = os.stat(race_file_path)
        file_sig = (int(st.st_mtime_ns), int(st.st_size))
    except Exception:
        file_sig = (0, 0)
    return (
        _resolved_path_for_cache(db_path),
        _resolved_path_for_cache(race_file_path),
        file_sig,
    )


class PredictionPipelineV4:
    def __init__(self, db_path="greyhound_racing_data.db"):
        # Resolve database path intelligently
        resolved_db = os.getenv("GREYHOUND_DB_PATH") or db_path
        candidates = [
            resolved_db,
            (
                os.path.join(".", resolved_db)
                if not os.path.isabs(resolved_db)
                else resolved_db
            ),
            os.path.join(".", "greyhound_racing_data.db"),
            os.path.join(".", "databases", "comprehensive_greyhound_data.db"),
            os.path.join(".", "databases", "greyhound_racing_data.db"),
        ]
        chosen = None
        for cand in candidates:
            try:
                if cand and os.path.isfile(cand):
                    chosen = cand
                    break
            except Exception:
                continue
        self.db_path = chosen or resolved_db
        if not os.path.isfile(self.db_path):
            logger.warning(
                f"⚠️ Database not found at {self.db_path}. Historical features may be empty. Set GREYHOUND_DB_PATH to fix."
            )
        else:
            logger.info(f"🗄️ Using database: {self.db_path}")

        self.ml_system_v4 = _get_cached_ml_system_v4(self.db_path)
        logger.info("🚀 Prediction Pipeline V4 - Advanced System Initialized")

    def predict_race_file(
        self, race_file_path: str, tgr_enabled: bool | None = None, optimizer_enabled: bool | None = None
    ) -> dict:
        """Main prediction method using ML System V4.

        Args:
            race_file_path: Path to upcoming race CSV
            tgr_enabled: Optional runtime toggle to include TGR features (DB-only) in predictions
        """
        logger.info(
            f"🚀 Starting prediction for: {os.path.basename(race_file_path)} using ML System V4"
        )

        # Pre-prediction module sanity check
        # IMPORTANT: This call ensures no historical-data collectors, scrapers, or heavy frameworks
        # are imported during prediction. Module guard enforces prediction_only import policy.
        # Keep this import local and NEVER at module top-level to avoid false positives in tests.
        try:
            if os.getenv("V4_SKIP_MODULE_GUARD", "0").strip().lower() not in ("1", "true", "yes"):
                # Import from package path to avoid being shadowed by scripts/utils.py when running scripts/*
                from utils.module_guard import pre_prediction_sanity_check as _mg_check
                _mg_check(
                    context="PredictionPipelineV4.predict_race_file",
                    extra_info={"race_file_path": os.path.basename(race_file_path)},
                )
        except Exception as e:
            logger.error(f"🛑 Module guard blocked prediction: {e}")
            # Provide clear, actionable error response
            guidance = []
            if hasattr(e, "resolution"):
                guidance = getattr(e, "resolution", [])
            return {
                "success": False,
                "error": str(e),
                "race_id": os.path.basename(race_file_path).replace(".csv", ""),
                "fallback_reason": "Disallowed module(s) loaded – see guidance",
                "resolution": guidance,
            }

        try:
            # Use CSV ingestion to read race data (prediction-only safe component)
            # NOTE: CsvIngestion reads a single race CSV (race data) and does not load
            # historical results. Keep ingestion imports narrow to avoid pulling in
            # broader scraping frameworks at prediction time.
            ingestion = CsvIngestion(race_file_path)
            parsed_race, validation_report = ingestion.parse_csv()

            if not validation_report.is_valid:
                errors = "\n".join(validation_report.errors)
                logger.error(
                    f"🛑 Validation failed for {race_file_path} with errors: {errors}"
                )
                return {"success": False, "error": errors, "race_id": race_file_path}

            # Convert to DataFrame
            race_data = pd.DataFrame(parsed_race.records, columns=parsed_race.headers)
            race_id = os.path.basename(race_file_path).replace(
                ".csv", ""
            )  # Use full filename without extension as race_id

            # Optionally persist embedded history into DB so DB-backed features can use it.
            # Keep this opt-in: prediction should not mutate restored/canonical DBs by default.
            try:
                ingest_mode = os.getenv("INGEST_EMBEDDED_HISTORY_ON_PREDICT", "0").strip().lower()
                ingest_on_predict = ingest_mode in ("1", "true", "yes", "on", "always", "force")
                force_ingest = ingest_mode in ("always", "force")
            except Exception:
                ingest_on_predict = False
                force_ingest = False

            # Default TGR toggle from env if not provided
            try:
                if tgr_enabled is None:
                    _tgr_env = os.getenv("TGR_FEATURES_ENABLED")
                    if _tgr_env is not None:
                        tgr_enabled = str(_tgr_env).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                pass
            if ingest_on_predict:
                try:
                    ingest_key = _embedded_history_ingest_key(self.db_path, race_file_path)
                    do_ingest = True
                    if not force_ingest:
                        with _EMBEDDED_HISTORY_INGEST_LOCK:
                            already_ingested = ingest_key in _EMBEDDED_HISTORY_INGESTED_KEYS
                        if already_ingested:
                            logger.debug(
                                "Embedded history ingestion skipped: already ingested for this process/file signature"
                            )
                            do_ingest = False

                    if do_ingest:
                        # Import lazily to avoid overhead when disabled
                        from scripts.ingest_embedded_form_history import upsert_embedded_history_and_meta as _ingest_hist

                        stats = _ingest_hist(self.db_path, race_file_path)
                        with _EMBEDDED_HISTORY_INGEST_LOCK:
                            _EMBEDDED_HISTORY_INGESTED_KEYS.add(ingest_key)
                        try:
                            logger.info(
                                f"🗄️ Embedded history ingested: inserted={stats.get('inserted')} skipped={stats.get('skipped')} into DB={self.db_path}"
                            )
                        except Exception:
                            pass
                except Exception as _ie:
                    logger.debug(f"Embedded history ingestion skipped/failed: {_ie}")

            # Map CSV columns to expected ML System V4 format
            race_data = self._map_csv_to_v4_format(race_data, race_file_path)
            race_data, dropped_leakage_fields = strip_target_leakage_columns(
                race_data, allow_labels=False
            )

            # Apply runtime TGR toggle if provided
            try:
                if tgr_enabled is not None and hasattr(
                    self.ml_system_v4, "set_tgr_enabled"
                ):
                    self.ml_system_v4.set_tgr_enabled(bool(tgr_enabled))
            except Exception:
                pass

            # Prepare ML system for this call with per-request overrides
            ml_for_call = self.ml_system_v4

            # If optimizer toggle requested, ensure optimizer is integrated for this instance
            try:
                if optimizer_enabled is True and getattr(ml_for_call, "accuracy_optimizer", None) is None:
                    os.environ["V4_DISABLE_ACCURACY_OPTIMIZER"] = "0"
                    try:
                        ml_for_call._initialize_accuracy_optimizer()
                    except Exception:
                        pass
            except Exception:
                pass

            # If optimizer explicitly disabled, create a fresh MLSystemV4 instantiated with optimizer off
            try:
                if optimizer_enabled is False:
                    prev = os.environ.get("V4_DISABLE_ACCURACY_OPTIMIZER")
                    os.environ["V4_DISABLE_ACCURACY_OPTIMIZER"] = "1"
                    try:
                        ml_for_call = MLSystemV4(self.db_path)
                    finally:
                        # restore previous env setting
                        if prev is None:
                            try:
                                del os.environ["V4_DISABLE_ACCURACY_OPTIMIZER"]
                            except Exception:
                                pass
                        else:
                            os.environ["V4_DISABLE_ACCURACY_OPTIMIZER"] = prev
            except Exception:
                pass

            # Apply runtime TGR toggle on the chosen ML system
            try:
                if tgr_enabled is not None and hasattr(ml_for_call, "set_tgr_enabled"):
                    ml_for_call.set_tgr_enabled(bool(tgr_enabled))
            except Exception:
                pass

            # Load feature flags (YAML + env overrides)
            flags, flag_sources = load_flags()

            # Fetch live odds for this race from DB (win + place)
            win_odds: dict[str, float] = {}
            place_odds: dict[str, float] = {}
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.cursor()
                    # Win odds
                    cur.execute(
                        """
                        SELECT dog_clean_name, odds_decimal
                        FROM live_odds
                        WHERE race_id = ? AND market_type = 'win' AND (is_current = 1 OR is_current IS NULL)
                        """,
                        (race_id,),
                    )
                    for dog, odds in cur.fetchall():
                        try:
                            if dog:
                                _store_market_odds(win_odds, dog, odds)
                        except Exception:
                            continue
                    # Place odds (Top 3). Prefer topN-aware schema; if missing, fallback to schema without topN.
                    try:
                        cur.execute(
                            """
                            SELECT dog_clean_name, odds_decimal, topN
                            FROM live_odds
                            WHERE race_id = ? AND market_type IN ('place','top3') AND (topN = 3 OR topN IS NULL) AND (is_current = 1 OR is_current IS NULL)
                            """,
                            (race_id,),
                        )
                        rows = cur.fetchall()
                        for dog, odds, topn in rows:
                            try:
                                if dog and (topn == 3 or topn is None):
                                    _store_market_odds(place_odds, dog, odds)
                            except Exception:
                                continue
                    except sqlite3.OperationalError:
                        # Fallback: schema without topN column
                        cur.execute(
                            """
                            SELECT dog_clean_name, odds_decimal
                            FROM live_odds
                            WHERE race_id = ? AND market_type IN ('place','top3') AND (is_current = 1 OR is_current IS NULL)
                            """,
                            (race_id,),
                        )
                        rows = cur.fetchall()
                        for dog, odds in rows:
                            try:
                                if dog:
                                    _store_market_odds(place_odds, dog, odds)
                            except Exception:
                                continue
            except Exception as e:
                logger.warning(f"Odds join failed for race {race_id}: {e}")

            # Fallback: if odds were not found under the filename race_id, try resolving a canonical race_id
            try:
                def _parse_from_filename(rid: str):
                    import re as _re
                    m = _re.match(r"^\s*Race\s+(\d+)\s*-\s*(.+?)\s*-\s*(\d{4}-\d{2}-\d{2})\s*$", str(rid) or "", _re.IGNORECASE)
                    if m:
                        try:
                            return int(m.group(1)), str(m.group(2)).strip(), str(m.group(3)).strip()
                        except Exception:
                            return None, None, None
                    return None, None, None

                def _resolve_alt_race_id(conn, rid: str, df):
                    # Try parse from filename first
                    rnum, vlabel, ymd = _parse_from_filename(rid)
                    if (not rnum or not ymd) and hasattr(df, "__class__"):
                        try:
                            # Attempt from DataFrame columns when available
                            import pandas as _pd  # noqa: F401
                            if isinstance(df, _pd.DataFrame) and len(df) > 0:
                                if rnum is None and "race_number" in df.columns and not df["race_number"].isna().all():
                                    try:
                                        rnum = int(_pd.to_numeric(df["race_number"], errors="coerce").dropna().mode().iloc[0])
                                    except Exception:
                                        rnum = None
                                if not ymd and "race_date" in df.columns and not df["race_date"].isna().all():
                                    try:
                                        ymd = str(df["race_date"].dropna().astype(str).iloc[0])[:10]
                                    except Exception:
                                        ymd = None
                                if not vlabel and "venue" in df.columns and not df["venue"].isna().all():
                                    try:
                                        vlabel = str(df["venue"].dropna().astype(str).iloc[0])
                                    except Exception:
                                        vlabel = None
                        except Exception:
                            pass
                    if not (rnum and ymd):
                        return None
                    # Query race_metadata to find a race_id by date + race_number, prefer venue match when possible
                    try:
                        cur = conn.cursor()
                        cur.execute(
                            """
                            SELECT race_id, venue FROM race_metadata
                            WHERE race_date = ? AND CAST(race_number AS INTEGER) = ?
                            """,
                            (str(ymd), int(rnum)),
                        )
                        rows = cur.fetchall() or []
                        if not rows:
                            return None
                        if vlabel:
                            # choose the first whose venue string loosely matches label
                            lbl = str(vlabel).upper().replace(" ", "")
                            def _score(ven):
                                vv = str(ven or "").upper().replace(" ", "")
                                # direct contains either way has higher score
                                if lbl and (lbl in vv or vv in lbl):
                                    return 2
                                # first 3 chars match (e.g., BAL ~ BALLARAT)
                                if len(lbl) >= 3 and vv.startswith(lbl[:3]):
                                    return 1
                                return 0
                            rows.sort(key=lambda r: _score(r[1]), reverse=True)
                        # return best candidate race_id
                        return rows[0][0] if rows else None
                    except Exception:
                        return None

                if (not win_odds) or (not place_odds):
                    alt_id = _resolve_alt_race_id(conn, race_id, race_data)
                    if alt_id and alt_id != race_id:
                        try:
                            # Re-query with the resolved id
                            win_odds = {}
                            place_odds = {}
                            cur = conn.cursor()
                            cur.execute(
                                """
                                SELECT dog_clean_name, odds_decimal
                                FROM live_odds
                                WHERE race_id = ? AND market_type = 'win' AND (is_current = 1 OR is_current IS NULL)
                                """,
                                (alt_id,),
                            )
                            for dog, odds in cur.fetchall() or []:
                                try:
                                    if dog:
                                        _store_market_odds(win_odds, dog, odds)
                                except Exception:
                                    continue
                            # place/top3 odds
                            try:
                                cur.execute(
                                    """
                                    SELECT dog_clean_name, odds_decimal, topN
                                    FROM live_odds
                                    WHERE race_id = ? AND market_type IN ('place','top3') AND (topN = 3 OR topN IS NULL) AND (is_current = 1 OR is_current IS NULL)
                                    """,
                                    (alt_id,),
                                )
                                for dog, odds, _topn in cur.fetchall() or []:
                                    try:
                                        if dog:
                                            _store_market_odds(place_odds, dog, odds)
                                    except Exception:
                                        continue
                            except sqlite3.OperationalError:
                                cur.execute(
                                    """
                                    SELECT dog_clean_name, odds_decimal
                                    FROM live_odds
                                    WHERE race_id = ? AND market_type IN ('place','top3') AND (is_current = 1 OR is_current IS NULL)
                                    """,
                                    (alt_id,),
                                )
                                for dog, odds in cur.fetchall() or []:
                                    try:
                                        if dog:
                                            _store_market_odds(place_odds, dog, odds)
                                    except Exception:
                                        continue
                            # use alt_id for downstream odds-based EV
                            if (win_odds or place_odds):
                                logger.info(f"Resolved alt race_id for odds join: {race_id} -> {alt_id}")
                        except Exception as _fe:
                            logger.debug(f"alt race_id fallback failed: {_fe}")
            except Exception:
                pass

            # Perform prediction with V4 system (pass odds and flags)
            try:
                result = ml_for_call.predict_race(
                    race_data,
                    race_id,
                    market_odds=win_odds if win_odds else None,
                    market_place_odds=place_odds if place_odds else None,
                    flags=flags,
                )
            except TypeError:
                # Backward-compat: some enhance wrappers may not accept market_place_odds
                try:
                    result = self.ml_system_v4.predict_race(
                        race_data,
                        race_id,
                        market_odds=win_odds if win_odds else None,
                        flags=flags,
                    )
                except TypeError:
                    # Last resort: minimal signature
                    result = self.ml_system_v4.predict_race(race_data, race_id)

            # Enrich metadata and race context for UI/consumers
            try:
                if isinstance(result, dict) and result.get("success"):
                    result["leakage_audit"] = {
                        "status": "passed",
                        "dropped_target_fields": dropped_leakage_fields,
                    }
                    # Ensure optimizer flag is present for UI clarity (default False)
                    if result.get("optimizer_enabled") is None and result.get("optimization_applied") is None:
                        result["optimizer_enabled"] = False
                    # Predictor/methods/version defaults
                    result.setdefault("predictor_used", "PredictionPipelineV4")
                    if not result.get("prediction_methods_used"):
                        result["prediction_methods_used"] = ["ml_system"]
                    result.setdefault("analysis_version", "ML System V4")

                    # Normalize prediction item keys for frontend compatibility
                    try:
                        preds = (
                            result.get("predictions")
                            or result.get("enhanced_predictions")
                            or []
                        )
                        if isinstance(preds, list):
                            model_version = (
                                result.get("model_version")
                                or result.get("primary_model_id")
                                or (
                                    ",".join(str(m) for m in result.get("model_ids_used", []))
                                    if result.get("model_ids_used")
                                    else None
                                )
                            )
                            if not model_version:
                                try:
                                    model_info = getattr(ml_for_call, "model_info", {}) or {}
                                    model_version = (
                                        model_info.get("model_version")
                                        or model_info.get("model_id")
                                        or model_info.get("model_type")
                                    )
                                except Exception:
                                    model_version = None
                            model_version = model_version or "unknown"
                            result.setdefault("model_version", model_version)

                            for p in preds:
                                if isinstance(p, dict):
                                    if "dog_name" not in p and "dog_clean_name" in p:
                                        p["dog_name"] = p.get("dog_clean_name")
                                    if "name" not in p and "dog_clean_name" in p:
                                        p["name"] = p.get("dog_clean_name")
                                    # Normalize probability keys for UI consumers
                                    if (
                                        p.get("win_prob") is None
                                        and p.get("win_probability") is not None
                                    ):
                                        try:
                                            wp = float(p.get("win_probability"))
                                            p["win_prob"] = max(0.0, min(1.0, wp))
                                        except Exception:
                                            pass
                                    if (
                                        p.get("win_probability") is None
                                        and p.get("win_prob_norm") is not None
                                    ):
                                        try:
                                            wp2 = float(p.get("win_prob_norm"))
                                            p["win_probability"] = max(
                                                0.0, min(1.0, wp2)
                                            )
                                        except Exception:
                                            pass
                                    if (
                                        p.get("win_prob_norm") is None
                                        and p.get("win_probability") is not None
                                    ):
                                        try:
                                            p["win_prob_norm"] = max(
                                                0.0, min(1.0, float(p.get("win_probability")))
                                            )
                                        except Exception:
                                            pass
                                    if p.get("win_prob_raw") is None:
                                        try:
                                            p["win_prob_raw"] = float(
                                                p.get("win_prob_norm", p.get("win_probability", 0.0))
                                            )
                                        except Exception:
                                            p["win_prob_raw"] = None
                                    if p.get("confidence_score") is None:
                                        try:
                                            p["confidence_score"] = float(
                                                p.get("confidence", p.get("confidence_level", 0.0))
                                            )
                                        except Exception:
                                            p["confidence_score"] = None
                                    if p.get("ev_win") is None:
                                        try:
                                            odds_value = _lookup_market_odds(win_odds, p)
                                            if (
                                                odds_value is not None
                                                and p.get("win_prob_norm") is not None
                                            ):
                                                p["ev_win"] = float(
                                                    float(p["win_prob_norm"])
                                                    * float(odds_value)
                                                    - 1.0
                                                )
                                            else:
                                                p.setdefault("ev_win", None)
                                        except Exception:
                                            p.setdefault("ev_win", None)
                                    p.setdefault("model_version", model_version)
                            market_summary = _annotate_market_context(preds, win_odds)
                            if market_summary.get("market_odds_count"):
                                result["market_context"] = market_summary
                            if market_summary.get("large_disagreement_count", 0) > 0:
                                quality_warnings = result.setdefault("quality_warnings", [])
                                warning = {
                                    "code": "large_model_market_disagreement",
                                    "message": "One or more runners have large model-vs-market probability disagreement; ranking was not changed.",
                                    "count": market_summary.get("large_disagreement_count"),
                                    "threshold": market_summary.get(
                                        "large_disagreement_threshold"
                                    ),
                                }
                                if warning not in quality_warnings:
                                    quality_warnings.append(warning)

                            try:
                                ensemble_count = int(
                                    result.get("ensemble_models_used")
                                    or result.get("ensemble_models")
                                    or 0
                                )
                            except Exception:
                                ensemble_count = 0
                            if ensemble_count <= 1:
                                quality_warnings = result.setdefault("quality_warnings", [])
                                warning = {
                                    "code": "single_model_no_ensemble_agreement",
                                    "message": "Only one model contributed, so model_agreement is not ensemble evidence.",
                                }
                                if warning not in quality_warnings:
                                    quality_warnings.append(warning)
                                for p in preds:
                                    if isinstance(p, dict):
                                        p.setdefault(
                                            "model_agreement_basis",
                                            "not_applicable_single_model",
                                        )
                                        _append_quality_flag(
                                            p, "single_model_no_ensemble_agreement"
                                        )
                            try:
                                def _prediction_sort_probability(item) -> float:
                                    if not isinstance(item, dict):
                                        return 0.0
                                    for key in (
                                        "rank_sort_probability",
                                        "win_prob_norm_unrounded",
                                        "win_probability_unrounded_norm",
                                        "win_prob_norm",
                                        "win_probability",
                                        "win_prob",
                                    ):
                                        try:
                                            value = item.get(key)
                                            if value is not None:
                                                return float(value)
                                        except Exception:
                                            continue
                                    return 0.0

                                def _prediction_sort_name(item) -> str:
                                    if not isinstance(item, dict):
                                        return ""
                                    return _normalize_odds_name_key(
                                        item.get("dog_clean_name")
                                        or item.get("dog_name")
                                        or item.get("name")
                                        or ""
                                    )

                                preds.sort(
                                    key=lambda item: (
                                        -_prediction_sort_probability(item),
                                        _prediction_sort_name(item),
                                    )
                                )
                                for rank, p in enumerate(preds, start=1):
                                    if isinstance(p, dict):
                                        p["predicted_rank"] = rank
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Inject CSV-derived historical stats from the enriched input DataFrame so UI can fallback gracefully
                    try:
                        preds = (
                            result.get("predictions")
                            or result.get("enhanced_predictions")
                            or []
                        )
                        if isinstance(preds, list) and len(preds) > 0:
                            # Build a normalization helper and a lookup map from race_data
                            def _norm(s: str) -> str:
                                try:
                                    import re

                                    return re.sub(
                                        r"[^A-Za-z0-9]", "", (s or "").upper()
                                    )
                                except Exception:
                                    return (s or "").upper().replace(" ", "")

                            try:
                                # race_data in this scope is the already-enriched DataFrame
                                csv_cols = [
                                    "csv_historical_races",
                                    "csv_avg_finish_position",
                                    "csv_best_finish_position",
                                    "csv_recent_form",
                                    "csv_win_rate",
                                    "csv_place_rate",
                                    "csv_avg_time",
                                    "csv_best_time",
                                    "csv_prefixed_history_rows",
                                    "csv_blank_history_rows",
                                    "csv_historical_sources",
                                    "parser_context",
                                    "target_field_warning",
                                    "distance_source",
                                    "grade_source",
                                    "weight_source",
                                    "starting_price_source",
                                ]
                                lookup = {}
                                if (
                                    isinstance(race_data, pd.DataFrame)
                                    and "dog_clean_name" in race_data.columns
                                ):
                                    for _, row_df in race_data.iterrows():
                                        key = _norm(str(row_df.get("dog_clean_name")))
                                        if not key:
                                            continue
                                        entry = {}
                                        for c in csv_cols:
                                            if c in race_data.columns and pd.notna(
                                                row_df.get(c)
                                            ):
                                                entry[c] = row_df.get(c)
                                        if entry:
                                            lookup[key] = entry
                                # Merge into predictions
                                for p in preds:
                                    if not isinstance(p, dict):
                                        continue
                                    dn = (
                                        p.get("dog_clean_name")
                                        or p.get("dog_name")
                                        or p.get("name")
                                    )
                                    key = _norm(str(dn))
                                    if key in lookup:
                                        for k, v in lookup[key].items():
                                            try:
                                                # Unconditionally reflect enriched CSV stats into predictions
                                                p[k] = v
                                            except Exception:
                                                # Never fail enrichment merge
                                                pass
                                    # Ensure presence of csv_historical_races key for downstream UI logic
                                    if "csv_historical_races" not in p:
                                        p["csv_historical_races"] = 0
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Race context
                    try:
                        cols = set(race_data.columns)
                        rc_venue = (
                            str(race_data["venue"].iloc[0])
                            if "venue" in cols and len(race_data) > 0
                            else None
                        )
                        rc_date = (
                            str(race_data["race_date"].iloc[0])
                            if "race_date" in cols and len(race_data) > 0
                            else None
                        )
                        rc_distance = None
                        if "distance" in cols and len(race_data) > 0:
                            _dist = pd.to_numeric(
                                race_data["distance"], errors="coerce"
                            ).dropna()
                            if len(_dist) > 0:
                                try:
                                    rc_distance = (
                                        int(_dist.mode().iloc[0])
                                        if not _dist.mode().empty
                                        else int(_dist.iloc[0])
                                    )
                                except Exception:
                                    rc_distance = (
                                        float(_dist.mode().iloc[0])
                                        if not _dist.mode().empty
                                        else float(_dist.iloc[0])
                                    )
                        rc_grade = None
                        if "grade" in cols and len(race_data) > 0:
                            rc_grade = str(race_data["grade"].iloc[0])
                        target_field_sources = {}
                        for source_col, field_name in (
                            ("distance_source", "distance"),
                            ("grade_source", "grade"),
                            ("weight_source", "weight"),
                            ("starting_price_source", "starting_price"),
                        ):
                            try:
                                if source_col in cols and len(race_data) > 0:
                                    values = (
                                        race_data[source_col]
                                        .dropna()
                                        .astype(str)
                                        .loc[lambda s: s.str.strip() != ""]
                                        .unique()
                                        .tolist()
                                    )
                                    if values:
                                        target_field_sources[field_name] = values[0]
                            except Exception:
                                continue
                        target_field_warnings = []
                        try:
                            if "target_field_warning" in cols and len(race_data) > 0:
                                for value in race_data["target_field_warning"].dropna():
                                    for part in str(value).split(";"):
                                        part = part.strip()
                                        if part and part not in target_field_warnings:
                                            target_field_warnings.append(part)
                        except Exception:
                            target_field_warnings = []
                        parser_context = None
                        try:
                            if "parser_context" in cols and len(race_data) > 0:
                                parser_context = str(race_data["parser_context"].iloc[0])
                        except Exception:
                            parser_context = None
                        total_dogs = (
                            int(race_data["dog_clean_name"].nunique())
                            if "dog_clean_name" in cols
                            else int(len(race_data))
                        )

                        result.setdefault(
                            "race_context",
                            {
                                "filename": os.path.basename(race_file_path),
                                "venue": rc_venue,
                                "race_date": rc_date,
                                "distance": (
                                    f"{int(rc_distance)}m"
                                    if rc_distance is not None
                                    else None
                                ),
                                "grade": rc_grade,
                                "total_dogs": total_dogs,
                                "parser_context": parser_context,
                                "target_field_sources": target_field_sources,
                                "target_field_warnings": target_field_warnings,
                            },
                        )
                        if target_field_warnings:
                            quality_warnings = result.setdefault("quality_warnings", [])
                            warning = {
                                "code": "target_fields_not_from_race_card",
                                "message": "Some target race fields were defaulted or inferred because the CSV rows are embedded form history.",
                                "fields": target_field_sources,
                                "details": target_field_warnings,
                            }
                            if warning not in quality_warnings:
                                quality_warnings.append(warning)
                    except Exception:
                        # Soft-fail race_context enrichment
                        pass
            except Exception:
                pass

            if result.get("success"):
                # Persist predictions for monitoring consumption
                try:
                    out_dir = os.path.join("predictions")
                    os.makedirs(out_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
                    out_path = os.path.join(out_dir, f"{race_id}_{ts}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        import json as _json
                        _json.dump(result, f, indent=2)
                except Exception as _e:
                    logger.debug(f"Could not persist prediction for monitoring: {_e}")
                logger.info(f"✅ Prediction successful for {race_id}")
            else:
                logger.warning(
                    f"❌ Prediction failed for {race_id}: {result.get('error')}"
                )

            return result

        except Exception as e:
            logger.error(f"Error processing file {race_file_path}: {str(e)}")
            return {"success": False, "error": str(e), "race_id": race_file_path}

    def _map_csv_to_v4_format(
        self, race_data: pd.DataFrame, race_file_path: str
    ) -> pd.DataFrame:
        """Map CSV columns to the expected ML System V4 format with proper data type handling.

        IMPORTANT: This CSV format contains race participants followed by their historical data.
        - Participants have actual dog names (e.g., "2. Austrian Rose")
        - Historical data rows have empty dog names (shown as '""' in CSV)
        - We extract both participants AND their embedded historical data for enrichment.
        """

        # Extract race information from filename and first row
        filename = os.path.basename(race_file_path)

        # Parse race date from filename (e.g., "Race 1 - GOUL - 01 August 2025.csv" or ISO "Race 7 - MURR - 2025-08-24.csv")
        parts = filename.replace(".csv", "").split(" - ")
        race_date = datetime.now().strftime("%Y-%m-%d")
        venue = "Unknown"
        if len(parts) >= 3:
            date_part = parts[2]
            # Try multiple common formats
            for fmt in ("%d %B %Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
                try:
                    race_date = datetime.strptime(date_part, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            venue = parts[1] if len(parts) > 1 else "Unknown"

        # Create mapped DataFrame with required columns (build once per unique dog)
        mapped_data = []

        # Helper for numeric conversion
        def safe_float_convert(value, default=0.0):
            """Safely convert value to float with fallback."""
            try:
                if pd.isna(value) or value == "" or value is None:
                    return default
                return float(pd.to_numeric(value, errors="coerce"))
            except (ValueError, TypeError):
                return default

        def _row_value(row, key, default=None):
            try:
                if isinstance(row, dict):
                    value = row.get(key, default)
                elif hasattr(row, "get"):
                    value = row.get(key, default)
                else:
                    value = default
                if value is None or pd.isna(value) or str(value).strip() == "":
                    return default
                return value
            except Exception:
                return default

        def _first_non_empty(columns: tuple[str, ...]):
            for column in columns:
                if column not in race_data.columns:
                    continue
                try:
                    values = race_data[column].dropna()
                    for value in values:
                        if str(value).strip() != "":
                            return value, column
                except Exception:
                    continue
            return None, None

        def _mode_numeric(column: str) -> float | None:
            try:
                if column not in race_data.columns:
                    return None
                values = pd.to_numeric(race_data[column], errors="coerce").dropna()
                if values.empty:
                    return None
                counts = values.value_counts()
                return float(counts.index[0])
            except Exception:
                return None

        def _looks_like_embedded_form_history() -> bool:
            columns = {str(c).strip().upper() for c in race_data.columns}
            historical_columns = {
                "PLC",
                "TIME",
                "WIN",
                "BON",
                "MGN",
                "W/2G",
                "PIR",
                "SP",
                "DATE",
                "TRACK",
            }
            try:
                dog_names = race_data.get("Dog Name")
                has_prefixed_names = bool(
                    dog_names.astype(str).str.match(r"^\s*\d{1,2}\s*[\.\):-]").any()
                )
            except Exception:
                has_prefixed_names = False
            return has_prefixed_names and len(columns.intersection(historical_columns)) >= 4

        embedded_form_history = _looks_like_embedded_form_history()
        target_field_warnings: list[str] = []
        if embedded_form_history:
            target_field_warnings.append("embedded_form_history_detected")

        explicit_distance, explicit_distance_column = _first_non_empty(
            (
                "Race Distance",
                "race_distance",
                "target_distance",
                "current_race_distance",
                "Distance",
            )
        )
        race_level_distance = None
        race_level_distance_source = None
        if explicit_distance is not None:
            race_level_distance = safe_float_convert(explicit_distance, 500.0)
            race_level_distance_source = f"target_column:{explicit_distance_column}"
        elif embedded_form_history:
            inferred_distance = _mode_numeric("DIST")
            if inferred_distance is not None:
                target_field_warnings.append(
                    f"historical_form_distance_mode_available:{inferred_distance:g}"
                )
            race_level_distance = None
            race_level_distance_source = "default_missing_target"

        explicit_grade, explicit_grade_column = _first_non_empty(
            ("Race Grade", "race_grade", "target_grade", "current_race_grade", "Grade")
        )
        race_level_grade = None
        race_level_grade_source = None
        if explicit_grade is not None:
            race_level_grade = str(explicit_grade).upper()
            race_level_grade_source = f"target_column:{explicit_grade_column}"
        elif embedded_form_history:
            race_level_grade = "G5"
            race_level_grade_source = "default_missing_target"
            target_field_warnings.append("grade_defaulted_no_target_column")

        seen = set()  # normalized dog names we've emitted participants for
        order = []    # preserve first-seen order
        current_dog_norm = None
        box_by_dog: dict[str, int] = {}

        # First pass: determine unique participants and stable box numbers
        for _, row in race_data.iterrows():
            raw_name = str(row.get("Dog Name", "") or "").strip()
            has_prefix = _has_numeric_prefix(raw_name)
            norm_name = _normalize_dog_name_no_prefix(raw_name)

            if has_prefix:
                # New participant header row
                if norm_name in seen:
                    logger.debug(f"Duplicate participant header encountered for '{norm_name}', skipping")
                    current_dog_norm = norm_name  # still update context
                    continue
                # Extract box from prefix or BOX column
                box_val = _extract_box_from_name_or_row(raw_name, row)
                if box_val is None:
                    # Fallback to sequence if completely missing
                    box_val = len(seen) + 1
                box_by_dog[norm_name] = int(box_val)
                seen.add(norm_name)
                order.append(norm_name)
                current_dog_norm = norm_name
            else:
                # No numeric prefix; treat as continuation if it matches current dog or is blank
                if norm_name and norm_name != current_dog_norm:
                    # If name appears without prefix and differs from current, we only accept as new
                    # participant if it hasn't been seen and there is a BOX column with a plausible value.
                    if norm_name not in seen:
                        box_fallback = _extract_box_from_name_or_row(raw_name, row)
                        if box_fallback is not None:
                            box_by_dog[norm_name] = int(box_fallback)
                            seen.add(norm_name)
                            order.append(norm_name)
                            current_dog_norm = norm_name
                        else:
                            # Ambiguous unprefixed row; treat as historical for current context
                            logger.debug(f"Unprefixed name '{norm_name}' without BOX treated as history for '{current_dog_norm}'")
                    # else already seen -> history row
                else:
                    # Blank name or same as current -> history
                    pass

        if not order:
            logger.warning(f"No valid dog data found in {race_file_path}")
            return pd.DataFrame()

        # Second pass: emit mapped participant rows once per unique dog in first-seen order
        participant_count = len(order)
        for norm_name in order:
            # Find the first row corresponding to this dog to pull auxiliary columns
            first_row = None
            for _, row in race_data.iterrows():
                rn = _normalize_dog_name_no_prefix(str(row.get("Dog Name", "") or "").strip())
                if rn == norm_name:
                    first_row = row
                    break
            row = first_row if first_row is not None else {}

            if embedded_form_history:
                weight_value = 30.0
                weight_source = "default_missing_target"
                starting_price_value = 3.0
                starting_price_source = "default_missing_target"
                distance_value = (
                    race_level_distance if race_level_distance is not None else 500.0
                )
                distance_source = race_level_distance_source or "default_missing_target"
                grade_value = race_level_grade or "G5"
                grade_source = race_level_grade_source or "default_missing_target"
            else:
                weight_value = safe_float_convert(_row_value(row, "WGT"), 30.0)
                weight_source = "csv_row:WGT"
                starting_price_value = safe_float_convert(_row_value(row, "SP"), 3.0)
                starting_price_source = "csv_row:SP"
                distance_value = (
                    race_level_distance
                    if race_level_distance is not None
                    else safe_float_convert(_row_value(row, "DIST"), 500.0)
                )
                distance_source = race_level_distance_source or "csv_row:DIST"
                if race_level_grade is not None:
                    grade_value = race_level_grade
                    grade_source = race_level_grade_source or "target_column"
                else:
                    grade_value = str(_row_value(row, "G", "G5") or "G5").upper()
                    grade_source = "csv_row:G"

            mapped_row = {
                "race_id": filename.replace(".csv", ""),
                "dog_clean_name": norm_name,
                "box_number": int(box_by_dog.get(norm_name, order.index(norm_name) + 1)),
                "weight": weight_value,
                "weight_source": weight_source,
                "starting_price": starting_price_value,
                "starting_price_source": starting_price_source,
                "trainer_name": str(_row_value(row, "TRAINER", "Unknown") or "Unknown"),
                "venue": str(venue).upper().replace(" ", "_").replace("/", "_"),
                "grade": grade_value,
                "grade_source": grade_source,
                "track_condition": "Good",
                "weather": "Fine",
                "temperature": 20.0,
                "humidity": 60.0,
                "wind_speed": 10.0,
                "field_size": participant_count,
                "race_date": race_date,
                "race_time": "14:30",
                "distance": distance_value,
                "distance_source": distance_source,
                "margin": None,
                "individual_time": None,
                "finish_position": None,
                "performance_rating": safe_float_convert(_row_value(row, "PERF"), 0.0),
                "speed_rating": safe_float_convert(_row_value(row, "SPEED"), 0.0),
                "class_rating": safe_float_convert(_row_value(row, "CLASS"), 0.0),
                "parser_context": (
                    "embedded_form_history" if embedded_form_history else "target_card"
                ),
                "target_field_warning": ";".join(target_field_warnings),
            }
            mapped_data.append(mapped_row)

        # Create DataFrame and ensure proper data types
        result_df = pd.DataFrame(mapped_data)

        # Ensure numeric columns are properly typed
        numeric_columns = [
            "box_number",
            "weight",
            "starting_price",
            "temperature",
            "humidity",
            "wind_speed",
            "field_size",
            "distance",
            "performance_rating",
            "speed_rating",
            "class_rating",
        ]

        for col in numeric_columns:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors="coerce").fillna(
                    0.0
                )

        # Extract embedded historical data for each dog and attach as supplementary info
        result_df = self._enrich_with_csv_historical_data(result_df, race_data)

        logger.info(f"📋 Mapped {len(result_df)} dogs for ML System V4 prediction")
        return result_df

    def _enrich_with_csv_historical_data(
        self, participants_df: pd.DataFrame, raw_csv_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract embedded historical data from CSV and attach to participant records.

        This supplements the database historical lookup with recent form data directly from the CSV.
        """
        logger.info("🔍 Extracting embedded historical data from CSV...")

        # Parse embedded historical data structure
        csv_historical_data = {}
        csv_historical_source_counts: dict[str, dict[str, int]] = {}
        csv_history_rows_dropped_post_target: dict[str, int] = {}
        current_dog = None  # normalized name without prefix
        seen_header_for_dog: set[str] = set()
        target_race_date = None
        try:
            if "race_date" in participants_df.columns and not participants_df.empty:
                target_race_date = pd.to_datetime(
                    participants_df.iloc[0].get("race_date"), errors="coerce"
                )
                if pd.isna(target_race_date):
                    target_race_date = None
                else:
                    target_race_date = target_race_date.date()
        except Exception:
            target_race_date = None

        def _norm(name: str) -> str:
            return _normalize_dog_name_no_prefix(name)

        # Helpers for tolerant, case-insensitive extraction and numeric parsing
        import re as __re
        def _row_get_ci(_row, keys):
            try:
                if not isinstance(_row, dict):
                    _row = dict(_row)
                # Build case-insensitive map once per row
                ci_map = {str(k).strip().lower(): k for k in _row.keys()}
                for k in keys:
                    lk = str(k).strip().lower()
                    if lk in ci_map:
                        val = _row.get(ci_map[lk])
                        if val is not None and str(val).strip() != "":
                            return val
                return None
            except Exception:
                return None
        def _to_int_like(v):
            try:
                if v is None:
                    return None
                s = str(v)
                m = __re.search(r"(\d+)", s)
                return int(m.group(1)) if m else None
            except Exception:
                return None
        def _to_float_like(v):
            try:
                if v is None:
                    return None
                s = str(v)
                m = __re.search(r"(\d+(?:\.\d+)?)", s)
                return float(m.group(1)) if m else None
            except Exception:
                return None

        def _append_historical_race(dog_name: str, row, source: str) -> bool:
            try:
                # Case-insensitive header support with tolerant numeric parsing
                plc = _row_get_ci(
                    row,
                    ["PLC", "Plc", "Place", "Placing", "Position", "Finish"],
                )
                tim = _row_get_ci(row, ["TIME", "Time", "Race Time", "RaceTime"])  # may include 's'
                dist = _row_get_ci(row, ["DIST", "Distance"])  # may include 'm'
                mgn = _row_get_ci(row, ["MGN", "Margin", "Beaten Margin"])
                wgt = _row_get_ci(row, ["WGT", "Weight"])
                date_val = _row_get_ci(row, ["DATE", "Date", "race_date", "Race Date"])
                track_val = _row_get_ci(row, ["TRACK", "Track", "Venue", "venue"])
                if target_race_date is not None and date_val not in (None, ""):
                    parsed_history_date = pd.to_datetime(date_val, errors="coerce")
                    if not pd.isna(parsed_history_date) and parsed_history_date.date() >= target_race_date:
                        csv_history_rows_dropped_post_target[dog_name] = (
                            csv_history_rows_dropped_post_target.get(dog_name, 0) + 1
                        )
                        return False

                historical_race = {
                    "date": date_val or "",
                    "track": track_val or "",
                    "finish_position": _to_int_like(plc),
                    "time": _to_float_like(tim),
                    "distance": _to_int_like(dist),
                    "margin": _to_float_like(mgn),
                    "weight": _to_float_like(wgt),
                }

                if historical_race["finish_position"] is not None:
                    csv_historical_data.setdefault(dog_name, []).append(historical_race)
                    source_counts = csv_historical_source_counts.setdefault(dog_name, {})
                    source_counts[source] = source_counts.get(source, 0) + 1
                    return True
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping malformed historical row for {dog_name}: {e}")
            return False

        for _, row in raw_csv_data.iterrows():
            raw_name = str(row.get("Dog Name", "") or "").strip()
            has_prefix = _has_numeric_prefix(raw_name)
            norm_name = _norm(raw_name)

            is_history_row = False
            history_source = "history_row"

            if has_prefix:
                # Two possibilities due to forward-fill:
                # 1) First occurrence for this dog (true header)
                # 2) Forward-filled continuation row for the current dog (should be treated as history)
                if norm_name not in seen_header_for_dog:
                    # First header occurrence for this dog
                    current_dog = norm_name
                    seen_header_for_dog.add(norm_name)
                    if current_dog not in csv_historical_data:
                        csv_historical_data[current_dog] = []
                    # Expert-form CSVs use the first prefixed row as both runner
                    # header and latest visible form row. Record that source
                    # explicitly so it cannot be mistaken for target-race data.
                    _append_historical_race(current_dog, row, "prefixed_form_row")
                    continue
                else:
                    # Already saw a header for this dog. If we're still within the same dog's block,
                    # treat this forward-filled prefixed row as a history row.
                    if current_dog == norm_name:
                        history_source = "forward_filled_prefixed_row"
                        is_history_row = True
                    else:
                        # Switching context back to another dog we've seen; treat as header switch
                        current_dog = norm_name
                        # Ensure key exists
                        if current_dog not in csv_historical_data:
                            csv_historical_data[current_dog] = []
                        continue
            else:
                # History rows: blank name OR same normalized name as current without prefix
                is_blank = (raw_name == "" or raw_name == '""')
                same_dog_unprefixed = (
                    current_dog is not None and norm_name == current_dog and not has_prefix
                )
                if current_dog and (is_blank or same_dog_unprefixed):
                    history_source = (
                        "blank_continuation_row"
                        if is_blank
                        else "same_name_continuation_row"
                    )
                    is_history_row = True

            if is_history_row and current_dog:
                _append_historical_race(
                    current_dog,
                    row,
                    history_source,
                )

        # Calculate CSV-based historical features for each participant
        enriched_participants = []

        for _, participant in participants_df.iterrows():
            participant_dict = participant.to_dict()
            dog_name = participant_dict["dog_clean_name"]
            dropped_rows = csv_history_rows_dropped_post_target.get(dog_name, 0)
            if dropped_rows:
                participant_dict["csv_history_rows_dropped_post_target"] = dropped_rows

            if dog_name in csv_historical_data and csv_historical_data[dog_name]:
                history = csv_historical_data[dog_name]

                # Calculate basic statistics from CSV historical data
                positions = [
                    h["finish_position"]
                    for h in history
                    if h["finish_position"] is not None
                ]
                times = [h["time"] for h in history if h["time"] is not None]

                if positions:
                    source_counts = csv_historical_source_counts.get(dog_name, {})
                    participant_dict["csv_historical_races"] = len(positions)
                    participant_dict["csv_prefixed_history_rows"] = int(
                        source_counts.get("prefixed_form_row", 0)
                    )
                    participant_dict["csv_blank_history_rows"] = int(
                        source_counts.get("blank_continuation_row", 0)
                    )
                    participant_dict["csv_historical_sources"] = ",".join(
                        sorted(source_counts)
                    )
                    participant_dict["csv_avg_finish_position"] = sum(positions) / len(
                        positions
                    )
                    participant_dict["csv_best_finish_position"] = min(positions)
                    participant_dict["csv_recent_form"] = (
                        positions[0] if positions else None
                    )  # Most recent finish
                    participant_dict["csv_win_rate"] = len(
                        [p for p in positions if p == 1]
                    ) / len(positions)
                    participant_dict["csv_place_rate"] = len(
                        [p for p in positions if p <= 3]
                    ) / len(positions)

                    if times:
                        participant_dict["csv_avg_time"] = sum(times) / len(times)
                        participant_dict["csv_best_time"] = min(times)

                    logger.debug(
                        f"{dog_name}: Found {len(positions)} CSV races, avg finish: {participant_dict['csv_avg_finish_position']:.1f}"
                    )
                else:
                    logger.debug(
                        f"{dog_name}: No valid historical position data in CSV"
                    )
            else:
                # No CSV historical data for this dog
                participant_dict["csv_historical_races"] = 0
                participant_dict["csv_prefixed_history_rows"] = 0
                participant_dict["csv_blank_history_rows"] = 0
                participant_dict["csv_historical_sources"] = ""
                logger.debug(f"{dog_name}: No CSV historical data found")

            enriched_participants.append(participant_dict)

        enriched_df = pd.DataFrame(enriched_participants)

        # Log summary of CSV enrichment
        dogs_with_csv_history = len(
            [p for p in enriched_participants if p.get("csv_historical_races", 0) > 0]
        )
        logger.info(
            f"📊 CSV Historical Enrichment: {dogs_with_csv_history}/{len(enriched_participants)} dogs have embedded historical data"
        )

        return enriched_df
