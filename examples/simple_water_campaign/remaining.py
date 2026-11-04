#!/usr/bin/env python3
"""Run the four remaining bounded campaigns sequentially from a login session."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from campaign_support import atomic_json, events, read_json, source_identity

CAMPAIGNS = ('cohort', 'failures', 'pyseqm', 'mi250')


def run_remaining(root, seed, budget_start=None, campaigns=None):
    campaigns = tuple(CAMPAIGNS if campaigns is None else campaigns)
    if not campaigns or len(set(campaigns)) != len(campaigns) or not set(campaigns) <= set(CAMPAIGNS):
        raise ValueError('Select distinct supported campaigns')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'execution.json').exists():
            raise ValueError('Preserve this executed root and choose a fresh directory')
        origin = time.time() if budget_start is None else budget_start
        if origin < 0 or origin > time.time():
            raise ValueError('Budget origin must be a past timestamp')
        atomic_json(root / 'execution.json', dict(start=origin, launched=time.time(), campaigns=campaigns, seed=str(Path(seed).resolve()),
                                                source=source_identity()))
        outcomes = {}
        for campaign in campaigns:
            directory = root / campaign
            atomic_json(root / 'result.json', dict(status='running', current_campaign=campaign, campaigns=outcomes))
            command = [sys.executable, '-B', str(Path(__file__).with_name('run.py')),
                       '--run-root', str(directory), '--campaign', campaign]
            if campaign != 'pyseqm':
                command += ['--import-seed', str(Path(seed).resolve())]
            if campaign == 'cohort' and budget_start is not None:
                command += ['--budget-start', str(budget_start)]
            print(f'Starting {campaign}.', flush=True)
            with (root / f'{campaign}.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait()
                except BaseException:
                    os.killpg(process.pid, signal.SIGINT)
                    process.wait(timeout=300)
                    raise
            result = read_json(directory / 'result.json') if (directory / 'result.json').exists() else {
                'status': 'failed', 'error': f'Campaign exited {code}; see {campaign}.log'}
            if result['status'] == 'running' or code and result['status'] == 'passed':
                result = dict(status='failed', returncode=code, partial_report=result,
                              error=f'Campaign terminated without a final report; see {campaign}.log')
            outcomes[campaign] = result
            # No later stage may allocate resources after an unverified cleanup.
            for run in [directory / 'bootstrap', *sorted((directory / 'cases').glob('*'))]:
                jobs = [r for r in events(run) if r['kind'] == 'job_submitted']
                if jobs and (not (run / 'cleanup.json').exists() or not read_json(run / 'cleanup.json')['verified']):
                    atomic_json(root / 'result.json', dict(status='incomplete', campaigns=outcomes,
                                error=f'Unverified cleanup: {run}'))
                    raise RuntimeError(f'Unverified cleanup: {run}')
            print(f'{campaign}: {result["status"]}.', flush=True)
        dependencies = dict(
            trained_cell_checkpoint=dict(status='open', reason='No qualified trained cell-input checkpoint supplied; existing periodic mechanics fixture is untrained'),
            stability=dict(status='open', reason='No potential with a documented stability domain and acceptance bounds supplied'),
            mi350=dict(status='deferred', reason='User deferred MI350 testing; outside the current Increment 7 campaign schedule'))
        atomic_json(root / 'dependencies.json', dependencies)
        atomic_json(root / 'result.json', dict(status='incomplete',
            selected_campaigns_passed=all(r['status'] == 'passed' for r in outcomes.values()),
            runnable_campaigns_passed=set(campaigns) == set(CAMPAIGNS) and all(r['status'] == 'passed' for r in outcomes.values()),
            campaigns=outcomes, dependencies=dependencies,
            reason='Required scientific-input gates remain open; do not infer full Increment 7 completion'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--import-seed', type=Path, required=True)
    parser.add_argument('--budget-start', type=float, help='Preserve the original cohort budget when repairing an early attempt')
    parser.add_argument('--campaigns', nargs='+', choices=CAMPAIGNS, help='Run selected campaigns in the supplied order')
    args = parser.parse_args()
    run_remaining(args.run_root, args.import_seed, args.budget_start, args.campaigns)
