#!/usr/bin/env python3
"""
Generate leakage report using FeatureQualityAssessment utilities.
Outputs: audit/leakage_report.json
"""
import json
from pathlib import Path
from datetime import datetime

from feature_quality_leakage_assessment import FeatureQualityAssessment


def main():
    out_dir = Path("audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "leakage_report.json"

    fqa = FeatureQualityAssessment()
    df = fqa.load_comprehensive_data()
    feature_catalog = fqa.catalogue_all_features(df) if df is not None and len(df) > 0 else {}
    mi_scores = fqa.calculate_predictive_utility(df) if df is not None and len(df) > 0 else {}
    leakage = fqa.perform_leakage_scan(df) if df is not None and len(df) > 0 else {}

    report = {
        "timestamp": datetime.now().isoformat(),
        "records": int(len(df)) if df is not None else 0,
        "feature_catalog_summary": {
            "count": len(feature_catalog)
        },
        "leakage": leakage,
        "predictive_utility_top10": [
            {"feature": k, "mi": float(v)} for k, v in list(mi_scores.items())[:10]
        ],
    }

    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

