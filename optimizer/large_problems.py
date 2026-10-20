"""Versioned, reproducible dense application models; no stored solutions.

Every variant has n binary decisions. Coefficients are deterministic given
kind/variant/size/seed, and all pairs have interactions before QUBO conversion.
These are synthetic benchmark scenarios, not a promise about real workloads.
"""
import numpy as np

GENERATOR = 'metaquant-forge-dense-v1; optimizer.large_problems.build_large; numpy.default_rng(seed)'


def pairs(rng, n, signed=False):
    upper = np.triu(rng.integers(1, 11, size=(n, n), dtype=np.int16), 1)
    if signed:
        signs = 2*rng.integers(0, 2, size=(n, n), dtype=np.int8)-1
        upper *= signs
    return (upper+upper.T).astype(np.float64)


def build_large(spec):
    from .problems import Problem
    n=spec.size
    rng=np.random.default_rng(spec.seed)
    data={'generator':GENERATOR, 'pair_interactions':n*(n-1)//2,
          'model_version':1, 'constraints':'Each decision is binary; no additional hard constraints.'}
    offset=0.
    if spec.variant=='coupled':
        # Same-zone resource contention plus a soft squared-load imbalance.
        durations=rng.integers(1,31,n)
        w=pairs(rng,n)
        coefficient=1/32  # exactly representable, same across all sizes
        total=float(durations.sum())
        q=w+4*coefficient*np.outer(durations,durations)
        q[np.diag_indices(n)]-=w.sum(axis=1)+4*coefficient*total*durations
        offset=float(w.sum()/2+coefficient*total**2)
        data.update(durations=durations.tolist(),balance_coefficient=coefficient,
            objective='same-zone contention + (load_A-load_B)^2/32',
            model='Two equivalent zones. Every request is assigned once. Balance is a soft cost, not a hard equality.')
    elif spec.variant=='interactions':
        benefits=rng.integers(1,21,n)
        w=pairs(rng,n,signed=True)
        q=w/2
        q[np.diag_indices(n)]=-benefits
        data.update(benefits=benefits.tolist(),
            objective='sum(pair_cost_ij*x_i*x_j) - sum(benefit_i*x_i)',
            model='Habitat action selection with overlap costs (positive pairs) and ecological synergy gains (negative pairs). No budget, deadlines or forbidden pairs.')
    elif spec.variant=='two_pools':
        costs=rng.integers(1,81,(n,2))
        w=pairs(rng,n,signed=True)
        q=w.copy()
        q[np.diag_indices(n)]=-w.sum(axis=1)+costs[:,1]-costs[:,0]
        offset=float(np.maximum(w,0).sum()/2+costs[:,0].sum())
        data.update(costs=costs.tolist(),
            objective='handling cost + same-pool contention + cross-pool communication',
            model='Each shipment goes to one of two logistics hubs. Positive pairs charge for sharing a pool; negative pairs charge for separation. No capacity/deadline constraints. This is not one-worker/one-shipment linear assignment.')
    elif spec.variant=='dense':
        # Signed Ising interactions plus local biases, expressed as a binary Q.
        w=pairs(rng,n,signed=True)
        fields=rng.integers(-5,6,n)
        q=2*w
        q[np.diag_indices(n)]=2*fields-2*w.sum(axis=1)
        offset=float(w.sum()/2-fields.sum())
        data.update(fields=fields.tolist(),
            objective='sum(w_ij*s_i*s_j) + sum(h_i*s_i), s=2*x-1',
            model='Dense binary sensor-mode interaction benchmark. No unstated business constraints.')
    else:
        raise ValueError('Unsupported large problem variant')
    return Problem(spec,q,offset,data)
