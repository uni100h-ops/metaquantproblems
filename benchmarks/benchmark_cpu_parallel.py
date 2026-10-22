"""Measure real CPU calls with identical problems, seeds and search budgets.

Run from the project root with numpy, pydantic and dwave-samplers installed:
  python benchmarks/benchmark_cpu_parallel.py --rounds 2 --cpu-limit 4
This benchmark does not call Modal or Fireworks, and does not measure a GPU.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from importlib.metadata import version

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable] = '1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from optimizer.catalog import PRESETS
from optimizer.channel_engines import CPU_PROFILE, run_cpu
from optimizer.dense_engines import classical_dense
from optimizer.live import collect_trials, CPU_TRIAL_WORKERS
from optimizer.problems import Spec, build_problem


def solve_cpu(problem,seed):
    if problem.spec.kind != 'channels':
        return classical_dense(problem,seed)
    raw = run_cpu(problem.weights,CPU_PROFILE,seed,None,1)
    bits = ((np.asarray(raw['spins'],dtype=int)+1)//2).tolist()
    value = problem.energy(bits)
    if abs(value-raw['remaining_interference']) > 1e-6:
        raise RuntimeError('objective_verification_failed')
    return dict(bits=bits,energy=value,seed=seed,elapsed_ms=raw['total_ms'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--cpu-limit',type=int,default=4)
    parser.add_argument('--output',default='cpu-parallel-benchmark.json')
    args = parser.parse_args()
    if args.rounds < 1 or args.cpu_limit < 1:
        parser.error('rounds and cpu-limit must be positive')
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:args.cpu_limit])
    report = dict(scope='Local CPU-only measurement; no Modal or GPU timing',
                  platform=platform.platform(),python=platform.python_version(),
                  numpy=np.__version__,dwave_samplers=version('dwave-samplers'),
                  allowed_logical_cpus=sorted(os.sched_getaffinity(0))
                    if hasattr(os,'sched_getaffinity') else None,
                  cpu_limit_applied=hasattr(os,'sched_getaffinity'),
                  trials=3,workers=CPU_TRIAL_WORKERS,profile=CPU_PROFILE,
                  timing='Wall time of three complete CPU trials and independent checks; '
                         'problem generation and warm-up excluded; no cached solutions',cases=[])
    # Cover both native paths: radio-channel Ising and general dense QUBO.
    for category in ('channels','custom'):
        preset = next(p for p in PRESETS if p['category']==category)
        spec = Spec.model_validate(preset['spec'])
        problem = build_problem(spec)
        seed = 1000+(spec.seed%100000)*100  # Same formula as the Render gateway.
        small = build_problem(spec.model_copy(update={'size':4}))
        collect_trials(solve_cpu,small,seed,3,workers=CPU_TRIAL_WORKERS)
        case = dict(category=category,spec=spec.model_dump(exclude_none=True),
                    sha256=problem.digest,solver_seeds=list(range(seed,seed+3)),rounds=[])
        original = None
        for round_number in range(args.rounds):
            measurements = {}
            for workers in ((1,CPU_TRIAL_WORKERS) if round_number%2==0 else (CPU_TRIAL_WORKERS,1)):
                start = time.perf_counter()
                rows = collect_trials(solve_cpu,problem,seed,3,workers=workers)
                wall = time.perf_counter()-start
                solutions = [(r['seed'],r['bits'],r['energy']) for r in rows]
                if original is None:
                    original = solutions
                if solutions != original:
                    raise RuntimeError('Solutions or seed order differ from the sequential baseline')
                key = 'sequential_s' if workers==1 else 'parallel_s'
                measurements[key] = wall
                print(f'{category} round {round_number+1}: {key}={wall:.3f}',flush=True)
            measurements['speedup'] = measurements['sequential_s']/measurements['parallel_s']
            case['rounds'].append(measurements)
        case['median_sequential_s'] = statistics.median(r['sequential_s'] for r in case['rounds'])
        case['median_parallel_s'] = statistics.median(r['parallel_s'] for r in case['rounds'])
        case['speedup'] = case['median_sequential_s']/case['median_parallel_s']
        case['identical_bits_energies_and_seeds'] = True
        case['energies'] = [r[2] for r in original]
        report['cases'].append(case)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({c['category']:{k:c[k] for k in
        ('median_sequential_s','median_parallel_s','speedup')} for c in report['cases']},indent=2))


if __name__ == '__main__':
    main()
