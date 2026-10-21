"""The exact fast-profile computation paths from the benchmark baseline.
The web compares these two named profiles; it does not claim to run its full CPU audit.
"""
import time
import warnings
import numpy as np
from dwave.samplers.sa.simulated_annealing import simulated_annealing
from dwave.samplers.greedy.descent import steepest_gradient_descent
CPU_PROFILE = dict(name='CPU_SA_8x200', method='sa', reads=8, sweeps=200, parallel=False)
GPU_PROFILE = dict(name='SB_GPU_128x1000', agents=128, max_steps=1000, early_stopping=False, timeout=None)


def objective(w, spins):
    """Independent CPU check, original integer weights, exact at these sizes."""
    s = np.asarray(spins, dtype=np.float64)
    if s.shape != (len(w),) or not np.isin(s, [-1, 1]).all():
        raise ValueError('Asignación de canales inválida.')
    # Each edge counted twice in W. Same-channel penalty is (1+s_i*s_j)/2.
    cost = float((w.sum() + s @ w @ s) / 4)
    return cost


def cpu_prepare(w):
    n = len(w)
    i, j = np.triu_indices(n, 1)
    couplings = w[i, j] / 2
    # Vectorized equivalent of D-Wave's default beta range for this positive,
    # zero-field graph. Avoid repeated Python dictionary traversal of all edges.
    nonzero = np.where(w > 0, w / 2, np.inf)
    min_per_spin = nonzero.min(axis=1)
    min_field = min_per_spin.min()
    hot = np.log(2) / w.sum(axis=1).max()
    cold = np.log(np.count_nonzero(min_per_spin == min_field) / 0.01) / (2 * min_field)
    # The native Cython bindings copy these sequences into C++ vectors.
    # Lists avoid a Python/NumPy scalar conversion for every edge at every call.
    return ([0.0] * n, i.tolist(), j.tolist(), couplings.tolist(), (hot, cold))


def cpu_task(prepared, n, profile, reads, seed):
    h, i, j, couplings, beta_range = prepared
    rng = np.random.default_rng(seed)
    initial = np.ascontiguousarray(2 * rng.integers(0, 2, (reads, n), dtype=np.int8) - 1)
    if profile['method'] == 'sa':
        betas = np.geomspace(*beta_range, profile['sweeps'])
        states, energies = simulated_annealing(
            reads, h, i, j, couplings, 1, betas, seed, initial, True, 'Metropolis', None)
    else:
        states, energies, _ = steepest_gradient_descent(reads, h, i, j, couplings, initial, False)
    idx = int(np.argmin(energies))
    return np.asarray(states[idx], dtype=np.int8).copy(), float(energies[idx])


def run_cpu(w, profile, seed, pool, workers):
    started = time.perf_counter()
    prepared = cpu_prepare(w)
    ready = time.perf_counter()
    count = min(workers, profile['reads']) if profile['parallel'] else 1
    sizes = [profile['reads'] // count + int(k < profile['reads'] % count) for k in range(count)]
    if count == 1:
        results = [cpu_task(prepared, len(w), profile, sizes[0], seed)]
    else:
        futures = [pool.submit(cpu_task, prepared, len(w), profile, size, seed + k * 10007)
                   for k, size in enumerate(sizes)]
        results = [f.result() for f in futures]
    spins, native_energy = min(results, key=lambda item: item[1])
    solved = time.perf_counter()
    cost = objective(w, spins)
    if abs(cost - (native_energy + w.sum() / 4)) > 1e-6:
        raise RuntimeError('La energía CPU no coincide con la comprobación independiente.')
    ended = time.perf_counter()
    return dict(engine='CPU', profile=profile['name'], seed=seed, cpu_workers=count,
                prepare_ms=1000*(ready-started), solve_ms=1000*(solved-ready),
                verify_ms=1000*(ended-solved), total_ms=1000*(ended-started),
                remaining_interference=cost, spins=spins.tolist())


def run_gpu(w, profile, seed):
    import torch
    import simulated_bifurcation as sb
    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.cuda.manual_seed(seed)
    with torch.inference_mode(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        matrix = torch.as_tensor(w, device='cuda:0', dtype=torch.float32) / 4
        torch.cuda.synchronize()
        ready = time.perf_counter()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        spins, energy = sb.minimize(
            matrix, domain='spin', device='cuda:0', dtype=torch.float32,
            agents=profile['agents'], max_steps=profile['max_steps'],
            early_stopping=profile['early_stopping'], timeout=profile['timeout'],
            sampling_period=50, convergence_threshold=50, mode='ballistic',
            heated=False, best_only=True, verbose=False)
        if not matrix.is_cuda or not spins.is_cuda or not energy.is_cuda:
            raise RuntimeError('La matriz, la solución y su energía deben permanecer en GPU.')
        if not bool(torch.isin(spins, torch.tensor([-1., 1.], device='cuda:0')).all().item()):
            raise RuntimeError('La solución GPU no contiene únicamente -1/+1.')
        # GPU objective in float64, then only the final assignment crosses back.
        s64 = spins.to(torch.float64)
        gpu_cost = s64 @ matrix.to(torch.float64) @ s64 + float(w.sum() / 4)
        end.record()
        result_spins = spins.to(torch.int8).cpu().numpy().copy()
        result_cost = float(gpu_cost.item())
        torch.cuda.synchronize()
        solved = time.perf_counter()
        stream_ms = begin.elapsed_time(end)
        cost = objective(w, result_spins)
        if abs(cost - result_cost) > 1e-6:
            raise RuntimeError('La energía GPU no coincide con la comprobación independiente.')
        ended = time.perf_counter()
    return dict(engine='NVIDIA_GPU', profile=profile['name'], seed=seed, cpu_fallback=False,
                matrix_device=str(matrix.device), solution_device=str(spins.device),
                objective_device=str(gpu_cost.device), gpu_stream_ms=stream_ms,
                prepare_ms=1000*(ready-started), solve_ms=1000*(solved-ready),
                verify_ms=1000*(ended-solved), total_ms=1000*(ended-started),
                remaining_interference=cost, spins=result_spins.tolist(),
                warnings=[str(item.message) for item in caught])
