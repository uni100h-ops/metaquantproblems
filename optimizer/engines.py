"""NVIDIA CUDA Simulated Bifurcation vs a compiled classical CPU baseline."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import platform
import time
import warnings
from importlib.metadata import version
import numpy as np
import torch
import simulated_bifurcation as sb
from .problems import Problem, exact_reference
from .channel_engines import run_gpu, CPU_PROFILE, GPU_PROFILE
from .cpu_engines import classical, channel_result, AGENTS, SA_SWEEPS
from .live import REPEATS

THREADS = 1  # Match the independent notebook's numerical-library configuration.
torch.set_num_threads(THREADS)
SB_MAX_STEPS = 10000
SB_TIMEOUT = 3.0
_gpu_verified = False


def device_name():
    if os.getenv('SB_DEVICE', 'cuda:0').lower() not in ('cuda', 'cuda:0'):
        raise RuntimeError('nvidia_gpu_required')
    if not torch.version.cuda or torch.version.hip:
        raise RuntimeError('cuda_required')
    if not torch.cuda.is_available():
        raise RuntimeError('gpu_unavailable')
    return 'cuda:0'


def verify_gpu():
    global _gpu_verified
    device = device_name()
    torch.cuda.set_device(0)
    with torch.inference_mode():
        a = torch.ones((4, 4), device=device)
        b = a @ a
        torch.cuda.synchronize(0)
        if not b.is_cuda or not bool(torch.all(b == 4).item()):
            raise RuntimeError('gpu_verification_failed')
    # Execute the actual package before publishing readiness.
    warmup()
    _gpu_verified = True
    return hardware()


def hardware():
    device = device_name()
    return {'sb_device': device, 'sb_device_name': torch.cuda.get_device_name(0),
            'sb_gpu_vendor': 'NVIDIA CUDA', 'cpu_fallback': False,
            'gpu_execution_verified': _gpu_verified,
            'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
            'gpu_architecture': getattr(torch.cuda.get_device_properties(0), 'gcnArchName', None),
            'traditional_device': 'CPU', 'cpu_architecture': platform.machine(),
            'torch_threads': THREADS, 'torch_version': str(torch.__version__),
            'cuda_version': torch.version.cuda, 'sb_version': version('simulated-bifurcation'),
            'classical_package': 'dwave-samplers', 'classical_version': version('dwave-samplers'),
            'numpy_version': np.__version__, 'python_version': platform.python_version()}


def sync():
    device_name()
    torch.cuda.synchronize(0)



def bifurcation(p: Problem, seed: int):
    device = device_name()
    if p.spec.kind == 'channels' and not p.spec.user_data():
        return channel_result(run_gpu(p.weights, GPU_PROFILE, seed), p)
    large=p.spec.variant!='standard'
    sync()
    start = time.perf_counter()
    torch.cuda.manual_seed(seed)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    with warnings.catch_warnings(record=True) as caught, torch.inference_mode():
        warnings.simplefilter('always')
        # The shared problem is uploaded once. Normalization, QUBO-to-Ising
        # conversion, dynamics, candidate selection and objective run on GPU.
        q_original = torch.as_tensor(p.q, dtype=torch.float64, device=device)
        scale = q_original.abs().amax()
        constant = bool((scale == 0).item())
        q = (q_original / scale.clamp_min(1e-12)).to(dtype=torch.float32)
        if constant:
            bits = torch.zeros(p.n, dtype=torch.float32, device=device)
        else:
            bits, _ = sb.minimize(q, domain='binary', dtype=torch.float32, device=device,
                                 agents=GPU_PROFILE['agents'] if large else AGENTS,
                                 max_steps=GPU_PROFILE['max_steps'] if large else SB_MAX_STEPS,
                                 mode='ballistic', heated=False, verbose=False,
                                 early_stopping=not large, sampling_period=50, convergence_threshold=50,
                                 timeout=None if large else SB_TIMEOUT)
        if q.device != torch.device(device) or bits.device != torch.device(device):
            raise RuntimeError('gpu_tensor_mismatch')
        if bits.shape != (p.n,) or not bool((torch.isfinite(bits) & ((bits == 0) | (bits == 1))).all().item()):
            raise RuntimeError('invalid_sb_solution')
        x64 = bits.to(dtype=torch.float64)
        energy = x64 @ q_original @ x64 + p.offset
        if not bool(torch.isfinite(energy).item()):
            raise RuntimeError('invalid_sb_solution')
        end.record()
        # Only final results cross back to CPU for JSON / independent validation.
        x = bits.to(dtype=torch.int64).cpu().tolist()
        value = float(energy.item())
    sync()
    if large and not np.isclose(value,p.energy(x),rtol=1e-9,atol=1e-6):
        raise RuntimeError('objective_verification_failed')
    warning_codes = sorted({type(w.message).__name__ for w in caught})
    return {'bits': x, 'elapsed_ms': (time.perf_counter()-start)*1000,
            'gpu_stream_ms': begin.elapsed_time(end), 'seed': seed,
            'energy': value, 'warnings': warning_codes,
            'matrix_device': str(q.device), 'solution_device': str(bits.device),
            'objective_device': str(energy.device), 'cpu_fallback': False,
            'degenerate_constant_objective': constant}


_warmed = False



def warmup():
    global _warmed
    if _warmed:
        return 0.0
    from .problems import Spec, build_problem
    start = time.perf_counter()
    p = build_problem(Spec(kind='channels',size=4,seed=3))
    classical(p, 3)
    bifurcation(p, 3)
    dense=build_problem(Spec(kind='custom',variant='dense',size=4,seed=3))
    classical(dense,3)
    bifurcation(dense,3)
    _warmed = True
    return (time.perf_counter()-start)*1000


def compare(p: Problem, seed=2026, progress=None, cancelled=None, partial=None):
    device_name()
    if not _gpu_verified:
        raise RuntimeError('gpu_verification_failed')
    cancelled = cancelled or (lambda: False)
    progress = progress or (lambda stage: None)
    started = time.perf_counter()
    if not _warmed:progress('warming_up')
    warmup_ms = warmup()
    progress('running')
    engines = {}
    # Both bundles start independently in warm, persistent processes.
    from .live import run_bundles, CPU_TRIAL_WORKERS
    partial_base = {'schema_version':'metaquant-forge-v1', 'problem':p.public(),
                    'hardware':hardware(), 'reference':{'known':False,'energy':None,
                    'method':'pending','elapsed_ms':0}, 'warmup_ms':warmup_ms,
                    'protocol':{'repetitions':REPEATS,'execution':'concurrent_processes'}}
    def engine_finished(name, engine):
        engines[name] = engine
        if partial:
            partial({**partial_base,'engines':{k:dict(v) for k,v in engines.items()},
                     'comparison_wall_ms':(time.perf_counter()-started)*1000,
                     'engine_errors':{},'complete':False})
    run_bundles(p,seed,engine_finished,cancelled)
    if cancelled(): raise RuntimeError('cancelled')
    progress('verifying')
    verify_start = time.perf_counter()
    reference = exact_reference(p)
    for engine in engines.values():
        gap=max(0.,engine['energy']-reference['energy']) if reference['known'] else None
        denominator=abs(reference['energy']) if reference['known'] else 0
        engine.update(optimality_gap_absolute=gap,
            optimality_gap_percent=100*gap/denominator if gap is not None and denominator>1e-9 else None,
            at_proven_optimum=bool(gap is not None and gap<1e-6))
    order = 'concurrent CPU/GPU processes; three CPU trials on three threads, three GPU trials sequentially; actual engine completion order recorded'
    reference['elapsed_ms'] = (time.perf_counter()-verify_start)*1000
    c, q = engines['traditional'], engines['quantum']
    ratio = c['bundle_wall_ms']/max(q['bundle_wall_ms'],1e-9)
    result = {'schema_version': 'metaquant-forge-v1', 'problem': p.public(), 'engines': engines,
            'reference': reference, 'hardware': hardware(), 'warmup_ms': warmup_ms,
            'comparison_wall_ms': (time.perf_counter()-started)*1000,
            'comparison': {'sb_speed_ratio': ratio, 'speed_ratio_basis':'bundle_wall_ms',
                           'sb_summed_call_ratio':c['solver_ms']/max(q['solver_ms'],1e-9),
                           'sb_energy_minus_traditional': q['energy']-c['energy'],
                           'both_feasible': c['decoded']['feasible'] and q['decoded']['feasible'],
                           'same_objective': abs(q['energy']-c['energy']) < 1e-6},
            'complete':True, 'protocol': {'execution':'concurrent_processes', 'repetitions': REPEATS, 'seed': seed, 'execution_order': order,
                         'performance_version':'2026-10-22-parallel-cpu-trials',
                         'traditional_trial_workers':min(CPU_TRIAL_WORKERS,REPEATS),
                         'traditional_workers_per_trial':1,'sb_trial_workers':1,
                         'candidates_per_run': AGENTS, 'traditional_sweeps': SA_SWEEPS,
                         'sb_max_steps': SB_MAX_STEPS, 'sb_timeout_seconds': SB_TIMEOUT,
                         'sb_mode': 'ballistic', 'sb_early_stopping': True,
                         'same_problem_sha256': p.digest, 'repair_or_local_polish': False,
                         'timing': 'Displayed time and speed ratio use bundle_wall_ms: elapsed time to complete all 3 trials, independent checks and decoding. CPU trials overlap. solver_ms is the sum of individual call durations, not CPU time or request latency. Warmup, reference, IPC, Meta API and HTTP are excluded; total request time is shown separately.',
                         'sb_compute': 'NVIDIA GPU: normalization, QUBO/Ising tensors, oscillator dynamics, candidate selection, binary validation and objective. Python control, input preparation, JSON and independent reference verification use CPU.',
                         'gpu_stream_timing': 'GPU event interval, including host dispatch gaps; not an isolated sum of kernel durations. The displayed comparison uses the complete three-trial bundle wall time for each engine.',
                         'scope': 'Configured heuristics, not equal stopping criteria or a claim against all classical solvers. Steps and sweeps are different units. Best solution across 3 runs; median also exported.'}}
    if (p.spec.kind == 'channels' and not p.spec.user_data()) or p.spec.variant!='standard':
        result['protocol'].update(
            candidates_per_run=None, traditional_candidates=8, sb_candidates=128,
            traditional_sweeps=200, traditional_workers=min(CPU_TRIAL_WORKERS,REPEATS),
            sb_max_steps=1000, sb_timeout_seconds=None, sb_early_stopping=False,
            traditional_profile=CPU_PROFILE, sb_profile=GPU_PROFILE,
            scope='CPU_SA_8x200 and SB_GPU_128x1000. All dense examples use these fixed profiles. Measure each new instance; no known optimum or universal speed advantage. The baseline is one configured CPU heuristic, not every classical solver.',
            sb_compute='NVIDIA GPU: QUBO/Ising conversion, dynamics, candidate selection and objective; host input preparation, Python control and independent checks use CPU. No CPU fallback.')
        result['comparison']['sb_objective_difference_percent'] = 100*(q['energy']-c['energy'])/abs(c['energy']) if c['energy'] else None
        result['comparison']['quality_reference'] = 'Displayed CPU profile only, not a known optimum or the best of the five CPU audit profiles.'
    if p.spec.kind == 'channels':
        result['comparison']['sb_objective_difference_percent'] = 100*(q['energy']-c['energy'])/c['energy'] if c['energy'] else None
        result['comparison']['quality_reference'] = 'Displayed CPU profile only; not a global optimum.'
    return result
