#!/usr/bin/env python3
import time
import threading
from datetime import datetime
from model_training_api import retrain_worker, training_jobs

def main():
    job_id = f"cli_test_{int(time.time())}"
    training_jobs[job_id] = {
        'id': job_id,
        'status': 'starting',
        'progress': 0,
        'model_id': 'comprehensive_training',
        'prediction_type': 'win',
        'training_data_days': 7,
        'force_retrain': True,
        'created_at': datetime.now().isoformat(),
        'started_at': datetime.now().isoformat(),
        'completed_at': None,
        'error_message': None,
        'thread': None,
    }
    print(f"Starting retrain_worker for job {job_id}...")
    th = threading.Thread(
        target=retrain_worker,
        args=(job_id, 'comprehensive_training', {'training_data_days': 7, 'force_retrain': True, 'prediction_type': 'win'}),
        daemon=True,
    )
    th.start()

    for i in range(120):
        j = training_jobs[job_id]
        print(f"[{i}] status={j['status']} progress={j['progress']} completed_at={j.get('completed_at')} err={j.get('error_message')}")
        if j['status'] in ('completed', 'failed'):
            break
        time.sleep(2)

    print('FINAL:', training_jobs[job_id])

if __name__ == '__main__':
    main()

