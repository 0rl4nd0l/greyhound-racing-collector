#!/usr/bin/env python3
import argparse
import logging

from ml_system_v4 import MLSystemV4

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Train win/place models (wrappers)")
    parser.add_argument("--mode", choices=["win", "place"], required=True, help="Model to train")
    parser.add_argument("--topN", type=int, default=3, help="Top-N for place model (default 3)")
    args = parser.parse_args()

    ml = MLSystemV4()

    # Use dedicated train entrypoints
    if args.mode == "win":
        logging.info("Training winner (Top 1) model...")
        ok = ml.train("win", topN=1)
    else:
        logging.info(f"Training placer (Top {args.topN}) model...")
        ok = ml.train("place", topN=args.topN)

    if not ok:
        raise SystemExit(1)

    logging.info("Training completed successfully")


if __name__ == "__main__":
    main()
