#!/usr/bin/env python3
"""Check finished E2 TinyFat array jobs on the cluster and sync valid outputs locally.

Run from the local workstation, not from TinyFat. The script:
1. asks squeue which array indices are still queued/running,
2. checks all other manifest indices on the cluster,
3. validates trials.h5 readability and row count,
4. reports missing/partial/unreadable/walltime/error cases,
5. rsyncs only complete readable run directories plus metadata/logs to a separate local folder.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

REMOTE_CHECK_CODE = r'''
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable


def expand_array_spec(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            left, right = part.split('-', 1)
            out.update(range(int(left), int(right) + 1))
        else:
            out.add(int(part))
    return out


def parse_squeue_indices(job_id: str) -> dict[int, str]:
    cmd = ['squeue.tinyfat', '-h', '-j', str(job_id), '-o', '%.200i %.40T']
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        # Some clusters return non-zero after a job leaves the queue. Treat as no active indices.
        return {}
    active: dict[int, str] = {}
    for line in proc.stdout.splitlines():
        fields = line.split(None, 1)
        if not fields:
            continue
        jid = fields[0].strip()
        state = fields[1].strip() if len(fields) > 1 else 'UNKNOWN'
        if jid == str(job_id):
            continue
        m_single = re.fullmatch(rf'{re.escape(str(job_id))}_(\d+)', jid)
        if m_single:
            active[int(m_single.group(1))] = state
            continue
        m_range = re.fullmatch(rf'{re.escape(str(job_id))}_\[(.+)\]', jid)
        if m_range:
            for idx in expand_array_spec(m_range.group(1)):
                active[idx] = state
    return active


def read_text(path: Path, limit: int = 200_000) -> str:
    if not path.exists():
        return ''
    data = path.read_text(encoding='utf-8', errors='replace')
    if len(data) > limit:
        return data[-limit:]
    return data


def log_state(log_root: Path, job_id: str, idx: int) -> tuple[str, list[str]]:
    paths = [Path(p) for p in glob.glob(str(log_root / f'slurm-*_{idx}.out'))]
    paths += [Path(p) for p in glob.glob(str(log_root / f'slurm-*_{idx}.err'))]
    text = '\n'.join(read_text(p) for p in paths)
    flags: list[str] = []
    if not paths:
        flags.append('no_logs')
    if '[E2] job=' in text:
        flags.append('runner_completed')
    if re.search(r'(?i)(time limit|timelimit|DUE TO TIME LIMIT|CANCELLED AT|cancelled)', text):
        flags.append('walltime_or_cancelled')
    if re.search(r'(?i)(traceback|exception|error:|segmentation fault|aborted|killed)', text):
        flags.append('error_in_logs')
    return text, flags


def h5_rows(path: Path) -> tuple[int | None, str | None]:
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            return int(f['trials']['trial_index'].shape[0]), None
    except Exception as exc:  # noqa: BLE001 - we want diagnostics for broken h5 files.
        return None, f'{type(exc).__name__}: {exc}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-id', required=True)
    ap.add_argument('--remote-root', required=True)
    ap.add_argument('--result-root', default='results/master_thesis/e2_tinyfat')
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--wires-json', default=None)
    args = ap.parse_args()

    remote_root = Path(args.remote_root)
    result_root = remote_root / args.result_root
    manifest_path = Path(args.manifest) if args.manifest else result_root / 'metadata/job_manifest_work.json'
    wires_path = Path(args.wires_json) if args.wires_json else result_root / 'metadata/wires.json'
    log_root = result_root / 'logs'

    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    jobs = payload['jobs']
    wires_payload = json.loads(wires_path.read_text(encoding='utf-8'))
    n_wires = len(wires_payload.get('wires', []))
    if n_wires <= 0:
        n_wires = 15

    active = parse_squeue_indices(args.job_id)
    all_indices = set(range(len(jobs)))
    finished_candidates = sorted(all_indices - set(active))

    records = []
    complete_readable = []
    for idx in finished_candidates:
        job = jobs[idx]
        outdir = Path(job['output_dir'])
        trial_count = int(job['config_spec']['trial_count'])
        expected = trial_count * n_wires
        h5_path = outdir / 'trials.h5'
        log_text, log_flags = log_state(log_root, args.job_id, idx)
        rows = None
        error = ''
        status = 'unknown'
        if not outdir.exists():
            status = 'missing_dir'
        elif not h5_path.exists():
            status = 'missing_h5'
        else:
            rows, err = h5_rows(h5_path)
            if err:
                status = 'unreadable'
                error = err
            elif rows is not None and rows >= expected:
                status = 'complete_readable'
            else:
                status = 'partial_readable'
        if status == 'complete_readable' and 'runner_completed' in log_flags and 'walltime_or_cancelled' not in log_flags and 'error_in_logs' not in log_flags:
            complete_readable.append(idx)
        records.append({
            'idx': idx,
            'job_name': job.get('job_name'),
            'status': status,
            'rows': rows,
            'expected_rows': expected,
            'log_flags': log_flags,
            'error': error,
            'output_dir': str(outdir),
            'relative_run_dir': str(outdir.relative_to(result_root)),
        })

    counts: dict[str, int] = {}
    for rec in records:
        counts[rec['status']] = counts.get(rec['status'], 0) + 1

    print(json.dumps({
        'job_id': args.job_id,
        'result_root': str(result_root),
        'manifest': str(manifest_path),
        'n_jobs': len(jobs),
        'n_wires': n_wires,
        'active_count': len(active),
        'finished_candidate_count': len(finished_candidates),
        'sync_complete_readable_count': len(complete_readable),
        'counts': counts,
        'active_min': min(active) if active else None,
        'active_max': max(active) if active else None,
    }, sort_keys=True))
    for rec in records:
        if rec['status'] != 'complete_readable' or set(rec['log_flags']) - {'runner_completed'}:
            print(json.dumps(rec, sort_keys=True))

    print('SYNC_INDEX_LIST=' + ','.join(str(i) for i in complete_readable))
    print('SYNC_RELATIVE_PATHS')
    for idx in complete_readable:
        print(records[finished_candidates.index(idx)]['relative_run_dir'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
'''


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--remote', default='iwhr106h@tinyx.nhr.fau.de')
    ap.add_argument('--remote-root', default='/home/woody/iwhr/iwhr106h/master-project')
    ap.add_argument('--remote-python', default='/home/woody/iwhr/iwhr106h/conda/envs/master-project/bin/python')
    ap.add_argument('--result-root', default='results/master_thesis/e2_tinyfat')
    ap.add_argument('--job-id', default='1192900')
    ap.add_argument('--local-output-root', type=Path, default=None)
    ap.add_argument('--dry-run', action='store_true', help='Only inspect; do not rsync.')
    return ap.parse_args()


def run(cmd: list[str], *, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print('+ ' + ' '.join(shlex.quote(part) for part in cmd), flush=True)
    return subprocess.run(cmd, input=input_text, text=True, check=check)


def remote_check(args: argparse.Namespace) -> tuple[list[str], Path]:
    remote_cmd = (
        f"cd {shlex.quote(args.remote_root)} && "
        f"{shlex.quote(args.remote_python)} - --job-id {shlex.quote(str(args.job_id))} "
        f"--remote-root {shlex.quote(args.remote_root)} "
        f"--result-root {shlex.quote(args.result_root)}"
    )
    proc = subprocess.run(
        ['ssh', args.remote, remote_cmd],
        input=REMOTE_CHECK_CODE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    print(proc.stdout, end='')
    lines = proc.stdout.splitlines()
    try:
        marker = lines.index('SYNC_RELATIVE_PATHS')
    except ValueError:
        raise RuntimeError('remote check did not print SYNC_RELATIVE_PATHS') from None
    rel_paths = [line.strip() for line in lines[marker + 1 :] if line.strip()]
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    local_root = args.local_output_root or Path(f'results/master_thesis/e2_tinyfat_finished_sync_{stamp}')
    return rel_paths, local_root


def write_rsync_file(paths: Iterable[str], local_root: Path) -> Path:
    tmp = Path('/tmp') / f'e2_finished_rsync_{os.getpid()}.txt'
    with tmp.open('w', encoding='utf-8') as f:
        f.write('/metadata/***\n')
        f.write('/logs/***\n')
        for rel in paths:
            f.write('/' + rel.rstrip('/') + '/***\n')
    print(f'rsync include list: {tmp}')
    print(f'local destination: {local_root}')
    return tmp


def main() -> int:
    args = parse_args()
    rel_paths, local_root = remote_check(args)
    print(f'complete readable run dirs to sync: {len(rel_paths)}')
    if args.dry_run:
        return 0
    if not rel_paths:
        print('No complete readable run dirs found; skipping rsync.')
        return 0
    include_file = write_rsync_file(rel_paths, local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    remote_src = f'{args.remote}:{args.remote_root}/{args.result_root}/'
    run([
        'rsync', '-av', '--partial', '--info=progress2',
        '--include-from', str(include_file),
        '--exclude', '*',
        remote_src,
        str(local_root) + '/',
    ])
    print(f'Synced complete readable E2 outputs to {local_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
