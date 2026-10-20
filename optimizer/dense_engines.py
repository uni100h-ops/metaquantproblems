"""Vectorized QUBO preparation for the native compiled CPU sampler.

No Python dictionaries over millions of edges; equivalent Ising coefficients
and default temperature bounds to dwave-samplers 1.8.0. Both engines receive Q.
"""
import time
import numpy as np
from .channel_engines import CPU_PROFILE, cpu_task


def prepare_qubo(q):
    n=len(q)
    fields=q.sum(axis=1)/2
    i,j=np.triu_indices(n,1)
    couplings=q[i,j]/2
    keep=couplings!=0
    i,j,couplings=i[keep],j[keep],couplings[keep]
    absolute=np.abs(q)/2
    np.fill_diagonal(absolute,0)
    max_field=float((absolute.sum(axis=1)+np.abs(fields)).max())
    absolute[absolute==0]=np.inf
    minimum=np.minimum(absolute.min(axis=1),np.where(fields!=0,np.abs(fields),np.inf))
    finite=minimum[np.isfinite(minimum)]
    if len(finite):
        smallest=float(finite.min())
        hot=np.log(2)/(2*max_field) if max_field else 1.
        cold=np.log(np.count_nonzero(finite==smallest)/0.01)/(2*smallest)
        beta_range=(hot,cold)
    else:
        beta_range=(0.1,1.)
    return fields.tolist(),i.tolist(),j.tolist(),couplings.tolist(),beta_range


def classical_dense(p,seed):
    start=time.perf_counter()
    prepared=prepare_qubo(p.q)
    ready=time.perf_counter()
    spins,native=cpu_task(prepared,p.n,CPU_PROFILE,CPU_PROFILE['reads'],seed)
    solved=time.perf_counter()
    bits=((spins+1)//2).astype(int).tolist()
    value=p.energy(bits)
    constant=float((p.q.sum()+np.trace(p.q))/4+p.offset)
    if not np.isclose(value,native+constant,rtol=1e-9,atol=1e-6):
        raise RuntimeError('objective_verification_failed')
    ended=time.perf_counter()
    return dict(bits=bits,seed=seed,energy=value,elapsed_ms=1000*(ended-start),
        prepare_ms=1000*(ready-start),solve_ms=1000*(solved-ready),
        verify_ms=1000*(ended-solved),profile=CPU_PROFILE['name'],cpu_workers=1,
        independent_cpu_energy=value,warnings=[])
