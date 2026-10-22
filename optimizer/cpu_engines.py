"""CPU solvers and shared result validation; no Torch or CUDA imports."""
import time
import numpy as np
import dimod
from dwave.samplers import SimulatedAnnealingSampler
from .problems import Problem
from .channel_engines import run_cpu, CPU_PROFILE

AGENTS = 32
SA_SWEEPS = 1000


def classical(p: Problem, seed: int):
    if p.spec.variant!='standard':
        from .dense_engines import classical_dense
        return classical_dense(p,seed)
    if p.spec.kind == 'channels' and not p.spec.user_data():
        raw = run_cpu(p.weights, CPU_PROFILE, seed, None, 1)
        return channel_result(raw, p)
    start = time.perf_counter()
    # Full symmetric matrix convention -> upper-triangle binary quadratic model.
    linear = {i: float(p.q[i,i]) for i in range(p.n)}
    quadratic = {(i,j): float(p.q[i,j]+p.q[j,i]) for i in range(p.n)
                 for j in range(i+1,p.n) if p.q[i,j]+p.q[j,i] != 0}
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, p.offset, dimod.BINARY)
    samples = SimulatedAnnealingSampler().sample(bqm, num_reads=AGENTS, num_sweeps=SA_SWEEPS,
                                                  seed=seed, randomize_order=True)
    sample = samples.first.sample
    x = np.array([sample[i] for i in range(p.n)],dtype=int)
    return {'bits': x.tolist(), 'elapsed_ms': (time.perf_counter()-start)*1000,
            'seed': seed, 'energy': p.energy(x), 'warnings': []}



def channel_result(raw, p):
    bits = ((np.asarray(raw['spins'], dtype=int) + 1) // 2).tolist()
    value = p.energy(bits)
    if abs(value - raw['remaining_interference']) > 1e-6:
        raise RuntimeError('objective_verification_failed')
    return {**{k: v for k, v in raw.items() if k != 'spins'}, 'bits': bits,
            'energy': value, 'elapsed_ms': raw['total_ms'],
            'independent_cpu_energy': value, 'warnings': raw.get('warnings', [])}

