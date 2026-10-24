"""Bounded README benchmark using the public CLI and identical serialized inputs.

Run only on a CUDA host. Use a dedicated output directory: generated synthetic
input directories are removed after each size to bound disk use. Timings come
from iterations.json; the external GPU sampler adds no CUDA synchronization.
For comparisons across GPU models, generate inputs once and use --input-root
on every host: a CUDA random seed alone does not guarantee identical files.
Example: python tools/benchmark_readme.py --repo . --output results/readme \
         --gpus 1 2 4 --sizes 6 8 10 12 --real-sdp path/to/mixed_n10
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import statistics
import subprocess
import threading
import time

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--gpus', nargs='+', type=int, default=[1])
p.add_argument('--sizes', nargs='*', type=int, default=[6, 8, 10, 12])
p.add_argument('--iterations', type=int, default=6)
p.add_argument('--real-sdp', type=Path)
p.add_argument('--baseline', type=Path)
p.add_argument('--input-root', type=Path, help='Reuse canonical synth_n<size> directories; never regenerate or delete them')
p.add_argument('--skip-real', action='store_true')
a = p.parse_args()
if a.iterations < 3 or any(g < 1 for g in a.gpus):
    p.error('Use at least three iterations and positive GPU counts')
a.repo = a.repo.resolve(); a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
for key in list(env):
    if key.startswith('SDPB_GPU_') or key.startswith('SDPVIBE_'):
        env.pop(key)
report = {'runs': {}, 'inputs': {}, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}}
report_path = a.output / 'matrix.json'
if report_path.exists():
    report = json.loads(report_path.read_text())
report.setdefault('invocations', []).append({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})

def save():
    temporary = report_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(report_path)

def run(command, log, timeout=2400, cwd=None):
    started = time.time()
    print(time.strftime('%H:%M:%S'), 'RUN', log.name, flush=True)
    with log.open('w') as f:
        proc = subprocess.Popen(command, cwd=cwd or a.repo, env=env,
                                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            raise
    if code:
        raise RuntimeError(f'{log}: return code {code}')
    return time.time() - started

def digest(directory):
    files = {}
    for f in sorted(directory.iterdir()):
        if f.suffix not in {'.bin', '.json'} or not f.is_file(): continue
        h = hashlib.sha256()
        with f.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b''): h.update(chunk)
        files[f.name] = {'sha256': h.hexdigest(), 'bytes': f.stat().st_size}
    return files

def solve(tag, directory, precision, gpus, real=False, repo=None):
    repo = repo or a.repo
    out = a.output / tag
    command = ['python', '-m', 'sdpvibe', '-s', str(directory), '-o', str(out),
               '-c', str(out) + '.ck', '--precision', str(precision), '--gpus', str(gpus),
               '--noFinalCheckpoint', '--verbosity', '2', '--writeSolution=y' if real else '--writeSolution=']
    if real:
        command += ['--findPrimalFeasible', '--findDualFeasible', '--detectPrimalFeasibleJump',
                    '--detectDualFeasibleJump', '--maxIterations', '500']
    else:
        command += ['--maxIterations', str(a.iterations)]
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted((repo / 'sdpvibe').glob('*.py'))}
    if tag in report['runs']:
        previous = report['runs'][tag]
        assert previous['command'] == command, f'Different command for existing run: {tag}'
        assert previous['source_sha256'] == hashes, f'Different source for existing run: {tag}'
        print('Already complete:', tag, flush=True); return
    if out.exists() or Path(str(out) + '.ck').exists():
        raise RuntimeError(f'Unrecorded output/checkpoint for {tag}; use a fresh output directory')
    stop = threading.Event(); samples = []
    def sample():
        while not stop.is_set():
            r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,temperature.gpu,power.draw,clocks.sm',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            samples.append({'elapsed': time.monotonic() - t0, 'gpu_csv': r.stdout.strip()})
            stop.wait(1)
    t0 = time.monotonic(); sampler = threading.Thread(target=sample, daemon=True); sampler.start()
    try:
        seconds = run(command, a.output / (tag + '.log'), cwd=repo)
    finally:
        stop.set(); sampler.join(timeout=10)
        (a.output / (tag + '.gpu.json')).write_text(json.dumps(samples) + '\n')
    rows = json.loads((out / 'iterations.json').read_text())
    times = [row['iter_time'] for row in rows]
    assert len(times) >= 3
    if not real: assert len(times) == a.iterations
    reason = next(line for line in (out / 'out.txt').read_text().splitlines() if 'terminateReason' in line)
    report['runs'][tag] = dict(command=command, wall_seconds=seconds, iterations=len(rows),
                               mean_seconds_iterations_2_plus=statistics.mean(times[1:]),
                               median_seconds_iterations_3_plus=statistics.median(times[2:]),
                               iteration_seconds=times, reason=reason,
                               source_sha256=hashes)
    save()
    print('DONE', tag, report['runs'][tag]['mean_seconds_iterations_2_plus'], flush=True)

if a.real_sdp and not a.skip_real:
    directory = a.real_sdp.resolve()
    manifest = digest(directory)
    if 'mixed_n10' in report['inputs']:
        assert report['inputs']['mixed_n10'] == manifest, 'Real input changed; use a new output directory'
    report['inputs']['mixed_n10'] = manifest; save()
    for g in a.gpus: solve(f'real_g{g}', directory, 768, g, real=True)

for n in a.sizes:
    directory = (a.input_root.resolve() if a.input_root else a.output) / f'synth_n{n}'
    if not (directory / 'control.json').exists():
        if a.input_root:
            p.error(f'Canonical input is missing: {directory}')
        run(['python', 'tools/synth_sdp.py', '--nmax', str(n), '--precision', '832',
             '--seed', '0', '--out', str(directory)], a.output / f'generate_n{n}.log')
    manifest = digest(directory)
    if f'synth_n{n}' in report['inputs']:
        assert report['inputs'][f'synth_n{n}'] == manifest, 'Synthetic input changed; use a new output directory'
    report['inputs'][f'synth_n{n}'] = manifest; save()
    # The first GPU count's parse populates the common binary-input cache.
    for g in a.gpus: solve(f'synth_n{n}_g{g}', directory, 832, g)
    if a.baseline and n == 10:
        for g in a.gpus: solve(f'baseline_n{n}_g{g}', directory, 832, g, repo=a.baseline.resolve())
    # Only generated, task-owned inputs are removed; hashes remain in matrix.json.
    if not a.input_root:
        shutil.rmtree(directory)
print('MATRIX COMPLETE', flush=True)
