#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)


def try_delegate_to_existing_trainer(model_glob: str, topN: int | None, output: str) -> bool:
    """Try to delegate to an existing backtesting script if available."""
    try:
        import importlib.util
        from pathlib import Path

        candidates = [
            Path("scripts/ml_backtesting_trainer.py"),
            Path("ml_backtesting_trainer.py"),
        ]
        for path in candidates:
            if path.exists():
                spec = importlib.util.spec_from_file_location("_bt_mod", str(path))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore
                    # Try to find a callable entrypoint
                    entry = getattr(mod, "main", None) or getattr(mod, "run", None)
                    if callable(entry):
                        logging.info(f"Delegating backtesting to {path}")
                        try:
                            if topN is not None:
                                entry(model_glob=model_glob, topN=topN, output=output)
                            else:
                                entry(model_glob=model_glob, output=output)
                            return True
                        except TypeError:
                            # Fallback: call with positional args
                            entry(model_glob, topN, output)
                            return True
        return False
    except Exception as e:
        logging.warning(f"Delegation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Backtest win/place models (wrapper)")
    parser.add_argument("--model", required=True, help="Glob for model artifact(s)")
    parser.add_argument("--topN", type=int, default=None, help="TopN for place backtest (use 3)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    # Try to delegate to existing trainer module if present
    if try_delegate_to_existing_trainer(args.model, args.topN, args.output):
        return

    # Minimal fallback output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "success": False,
            "error": "No integrated backtesting trainer found; wrapper did not execute a full backtest.",
            "model_glob": args.model,
            "topN": args.topN,
        }, f, indent=2)
    logging.warning("Backtesting wrapper wrote a fallback artifact; integrate with ml_backtesting_trainer.py for full results.")


if __name__ == "__main__":
    main()
