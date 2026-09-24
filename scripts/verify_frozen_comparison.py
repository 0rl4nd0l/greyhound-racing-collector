"""Verify a retained four-way comparison without opening target outcomes."""
import argparse
import json
from pathlib import Path
from src.predictor.future_comparison import verify_comparison

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--output-root",type=Path,required=True);p.add_argument("--admission",type=Path,required=True)
    p.add_argument("--expected-plan-sha256")
    a=p.parse_args();print(json.dumps(verify_comparison(a.output_root,a.admission,expected_plan_sha256=a.expected_plan_sha256),sort_keys=True))
