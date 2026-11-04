"""Render the user's current experiment table from durable run artifacts."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import time
import subprocess


def stage_status(root, *, labels=False):
    """Keep a started stage active across resource waits and resumptions."""
    root = Path(root)
    if any((root / name).exists() for name in ('COMPLETE.json', 'complete.json')):
        return 'completed'
    if labels:
        return 'generating' if any((root / name).exists() for name in (
            'contract.json', 'manifest.json')) else 'pending'
    # Contracts are written before preflight and before the first optimizer
    # update. Only durable checkpoints or actual train events prove training.
    if any((root / 'checkpoints' / name).exists() for name in ('latest.ckpt', 'latest.pt', 'latest.pkl')):
        return 'training'
    for name in ('metrics.jsonl', 'events.jsonl'):
        path = root / name
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # A concurrent append can leave a partial last line.
                    if event.get('event') == 'train' and event.get('global_step', event.get('step', 0)) > 0:
                        return 'training'
    if any((root / name).exists() for name in ('training_contract.json', 'contract.json')):
        return 'initializing'
    return 'pending'


def complete_predecessors(stages):
    """A later active/completed stage proves every prerequisite completed."""
    stages = list(stages)
    for index in range(len(stages) - 1, -1, -1):
        if stages[index] in {'completed', 'training', 'generating', 'evaluating'}:
            for previous in range(index):
                if not stages[previous].startswith('completed'):
                    stages[previous] = 'completed'
            break
    for index in range(1, len(stages)):
        if stages[index] == 'pending' and stages[index - 1].startswith('completed'):
            stages[index] = 'human'
    return stages


def render(registry, job_states=None):
    """Produce one status table and update time, without run logs."""
    job_states = job_states or {}
    failed_states = {'FAILED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'TIMEOUT', 'CANCELLED'}
    dp_runs = {run['route']: run for run in registry.get('dp_stage2_runs', [])}
    dp_sources = {
        'ManiFeel DP + UViP': None,
        'UniVTAC DP + UViP': '/data/wangzihao/experiments/univtac/l1_six_task_batch64_ep500_seed42_20261102_233553/train',
        'Tabero DP + UViP': '/data/wangzihao/experiments/tabero/l1_field_fs_batch64_ep200_seed42_20261102_234707/train',
    }
    selections = registry.get('dp_teacher_selections', {})
    rows = []
    for route, source in dp_sources.items():
        first = 'completed' if source is None or (Path(source) / 'COMPLETE.json').exists() else 'training'
        selection = 'completed' if route == 'ManiFeel DP + UViP' or route in selections else 'pending'
        labels = 'completed' if route == 'ManiFeel DP + UViP' else 'pending'
        train = evaluation = 'pending'
        run = dp_runs.get(route)
        if run:
            if run.get('labels'):
                labels = stage_status(run['labels'], labels=True)
                if labels == 'pending' and run.get('job_id'):
                    labels = 'queued'
            train = stage_status(run['output'])
            scheduler_state = job_states.get(str(run.get('job_id')), '')
            if scheduler_state in failed_states and train != 'completed':
                if labels == 'completed':
                    train = 'failed'
                else:
                    labels = 'failed'
            elif scheduler_state == 'PENDING' and labels == 'completed' and train in {'pending', 'initializing'}:
                train = 'queued'
            elif scheduler_state == 'RUNNING' and labels == 'completed' and train == 'pending':
                train = 'initializing'
            if run.get('evaluation_complete'):
                evaluation = 'completed'
        stages = complete_predecessors([first, selection, labels, train, evaluation])
        if route == 'ManiFeel DP + UViP':
            stages[1] = 'completed（E50 EMA）'
        elif route in selections:
            selected = selections[route]
            stages[1] = f"completed（E{selected['epoch']} {selected.get('weights', 'ema').upper()}）"
        rows.append([route, *stages])
    for run in registry.get('runs', []):
        root = Path(run['output'])
        route = ('UniVTAC FTP-1 / ' + run['task'] if run['model'] == 'ftp1' else 'Tabero-VTLA + UViP')
        labels = stage_status(root / 'labels', labels=True)
        training = stage_status(root)
        selection = 'completed' if (root / 'preflight/COMPLETE.json').exists() else 'pending'
        if job_states.get(str(run.get('job_id'))) in failed_states:
            if labels == 'completed' and training != 'completed':
                training = 'failed'
            elif labels != 'completed':
                labels = 'failed'
        rows.append([route, *complete_predecessors(['completed', selection, labels, training, 'pending'])])
    lines = [f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S %z')}", '',
             '| 路线 / 任务 | Stage 1 Teacher | Teacher Selection / 原生基线 | Utility Labels | Stage 2 Training | Stage 2 Evaluation |',
             '|---|---|---|---|---|---|']
    lines.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    return '\n'.join(lines) + '\n'


def update():
    path = os.environ.get('UVIP_RUN_REGISTRY')
    if not path:
        return
    registry_path = Path(path)
    with registry_path.with_suffix('.progress.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = json.loads(registry_path.read_text())
        document = Path(registry.get('summary_document') or registry['progress_document'])
        temporary = document.with_name(document.name + f'.{os.getpid()}.tmp')
        states = {}
        try:
            output = subprocess.check_output(['squeue', '--me', '-h', '-o', '%i|%T'], text=True, timeout=15)
            states.update(line.split('|', 1) for line in output.splitlines() if '|' in line)
            jobs = {str(r['job_id']) for r in registry.get('runs', []) + registry.get('dp_stage2_runs', []) if r.get('job_id')}
            missing = jobs - states.keys()
            if missing:
                output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(sorted(missing)),
                    '--format=JobIDRaw,State'], text=True, timeout=15)
                for line in output.splitlines():
                    fields = line.split('|')
                    if len(fields) >= 2:
                        states[fields[0]] = fields[1].split()[0].rstrip('+')
        except (subprocess.SubprocessError, OSError):
            pass
        temporary.write_text(render(registry, states))
        os.replace(temporary, document)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    while True:
        if args.watch:
            import runpy
            runpy.run_path(__file__)['update']()
        else:
            update()
        if not args.watch:
            break
        time.sleep(args.interval)
