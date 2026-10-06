"""Source-free speed worker. Live activation requires the allocation coordinator."""
import argparse
from pathlib import Path

from race_collection.prospective_speed_runtime import calculate, put_new, run_job
from race_collection.retained_card_timing_coverage import Reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('calculate', 'replay', 'prospective', 'tick', 'evaluate'))
    parser.add_argument('--job')
    parser.add_argument('--job-sha256')
    parser.add_argument('--output')
    parser.add_argument('--config')
    parser.add_argument('--config-sha256')
    args = parser.parse_args()
    if args.mode == 'tick':
        if not args.config or not args.config_sha256:
            parser.error('tick requires --config and --config-sha256')
        from race_collection.prospective_speed_coordinator import tick
        print(tick({'path': args.config, 'sha256': args.config_sha256})['status'])
        return
    if not args.job or not args.job_sha256 or not args.output:
        parser.error('execution requires --job, --job-sha256 and --output')
    reference = {'path': args.job, 'sha256': args.job_sha256}
    if args.mode == 'evaluate':
        from race_collection.prospective_speed_evaluation_io import run_evaluation
        print(run_evaluation(reference, args.output)['status'])
    elif args.mode == 'calculate':
        put_new(Path(args.output) / 'forecast.json', calculate(Reader().json(reference)))
    else:
        result = run_job(reference, args.output, replay=args.mode == 'replay')
        print(result['status'])


if __name__ == '__main__':
    main()
