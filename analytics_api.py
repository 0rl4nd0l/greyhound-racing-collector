import os
import glob
import json
import csv
from typing import Optional
from flask import Blueprint, jsonify, request, send_file

# Lightweight analytics blueprint for cohort and registry reports
analytics_bp = Blueprint("analytics_api", __name__)


def _find_latest(pattern_dir: str, pattern_glob: str) -> Optional[str]:
    try:
        pattern = os.path.join(pattern_dir, pattern_glob)
        files = glob.glob(pattern)
        return max(files, key=os.path.getmtime) if files else None
    except Exception:
        return None


@analytics_bp.get("/api/backtests/cohort/latest")
def api_backtests_cohort_latest():
    """Return the latest cohort backtest report JSON from ml_backtesting_results.

    Env overrides:
      - BACKTEST_RESULTS_DIR: directory to search (default ./ml_backtesting_results)

    Response: { success, filename?, updated_at?, cohort_report? }
    """
    try:
        results_dir = os.environ.get("BACKTEST_RESULTS_DIR") or os.path.join(
            os.getcwd(), "ml_backtesting_results"
        )
        latest = _find_latest(results_dir, "cohort_report_*.json")
        if not latest:
            return jsonify({"success": False, "error": "No cohort report found"}), 404
        with open(latest, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return jsonify(
            {
                "success": True,
                "filename": os.path.basename(latest),
                "updated_at": os.path.getmtime(latest),
                "cohort_report": payload,
            }
        )
    except Exception as e:
        return (
            jsonify({"success": False, "error": f"Failed to read cohort report: {e}"}),
            500,
        )


@analytics_bp.get("/api/registry/report")
def api_registry_report():
    """Return the latest registry CSV report as JSON (or raw CSV when format=csv).

    Env overrides:
      - MODEL_REGISTRY_DIR: directory to search (default ./model_registry)

    Query params:
      - format=csv to return the raw CSV file

    Response (JSON): { success, filename?, updated_at?, count?, rows? }
    """
    try:
        registry_dir = os.environ.get("MODEL_REGISTRY_DIR") or os.path.join(
            os.getcwd(), "model_registry"
        )
        latest = _find_latest(registry_dir, "registry_report_*.csv")
        if not latest:
            return jsonify({"success": False, "error": "No registry report found"}), 404

        if (request.args.get("format") or "").lower() == "csv":
            # Pass-through CSV
            return send_file(
                latest,
                mimetype="text/csv",
                as_attachment=False,
                download_name=os.path.basename(latest),
            )

        with open(latest, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return jsonify(
            {
                "success": True,
                "filename": os.path.basename(latest),
                "updated_at": os.path.getmtime(latest),
                "count": len(rows),
                "rows": rows,
            }
        )
    except Exception as e:
        return (
            jsonify({"success": False, "error": f"Failed to parse registry CSV: {e}"}),
            500,
        )

