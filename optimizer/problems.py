"""Deterministic, synthetic QUBOs; one canonical x.T @ Q @ x + offset for both engines."""
from dataclasses import dataclass
from hashlib import sha256
import json
import numpy as np
from .specs import Spec

from .catalog import PRESETS


@dataclass
class Problem:
    spec: Spec
    q: np.ndarray
    offset: float
    data: dict
    weights: np.ndarray | None = None

    @property
    def n(self):
        return len(self.q)

    @property
    def digest(self):
        if self.spec.kind == 'channels':
            return sha256(np.asarray(self.weights, dtype='<f8').tobytes()).hexdigest()
        if self.spec.variant!='standard':
            h=sha256()
            h.update(np.asarray(self.q,dtype='<f8').tobytes())
            h.update(np.asarray([self.offset],dtype='<f8').tobytes())
            return h.hexdigest()
        return sha256(json.dumps({'Q': self.q.tolist(), 'offset': self.offset}, sort_keys=True,
                                 separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def energy(self, x):
        x = np.asarray(x, dtype=float)
        return float(x @ self.q @ x + self.offset)

    def public(self):
        spec = self.spec.model_dump(exclude_none=True)
        if self.spec.kind == 'channels':
            return {'spec': spec, 'variables': self.n, 'data_source': 'user_supplied' if self.spec.user_data() else 'synthetic',
                    'Q': None, 'offset': self.offset, 'data': self.data,
                    'sha256': self.digest, 'hash_convention': 'SHA256 of W as little-endian float64, row-major; matches the standalone notebook',
                    'matrix_export': 'W is supplied in spec.weights.' if self.spec.user_data() else 'Reconstruct W from size, seed and generator; the large matrix is not sent through the web queue.',
                    'convention': 'minimize remaining same-session audience conflict: x.T @ Q @ x + offset; Q=W-diag(row_sums(W)); offset=sum(W)/2; session=x+1'}
        if self.spec.variant!='standard':
            compact={k:v for k,v in self.data.items() if k not in ('durations','benefits','costs','fields')}
            return {'spec':spec,'variables':self.n,'data_source':'synthetic','Q':None,
                'offset':self.offset,'data':compact,'sha256':self.digest,
                'hash_convention':'SHA256 of Q little-endian float64 row-major, then offset little-endian float64',
                'matrix_export':'Reconstruct with optimizer.problems.build_problem(Spec(**spec)) from this release; generator version and seed are recorded.',
                'convention':'minimize x.T @ Q @ x + offset; x in {0,1}^n'}
        if self.spec.kind == 'custom':
            spec.pop('size', None)
            spec.pop('seed', None)
        return {'spec': spec, 'variables': self.n,
                'data_source': 'user_supplied' if self.spec.user_data() else 'synthetic',
                'Q': self.q.tolist(), 'offset': self.offset, 'data': self.data,
                'sha256': self.digest, 'convention': 'minimize x.T @ Q @ x + offset; x in {0,1}^n'}

    def decode(self, bits):
        x = np.asarray(bits, dtype=int)
        if x.shape != (self.n,) or not np.isin(x, [0, 1]).all():
            raise ValueError('Invalid binary solution')
        chosen = (np.flatnonzero(x) + 1).tolist()
        other = (np.flatnonzero(1-x) + 1).tolist()
        kind, d = self.spec.kind, self.data
        metrics, violations = {}, []
        if kind == 'channels':
            remaining = self.energy(x)
            metrics = {'remaining_interference': remaining,
                       'avoided_interference': self.offset - remaining,
                       'channel_1_count': int(np.sum(x == 0)), 'channel_2_count': int(np.sum(x == 1))}
            chosen, other = other, chosen
        elif kind == 'maxcut':
            cut = sum(w for i,j,w in d['edges'] if x[i] != x[j])
            metrics = {'cut_weight': float(cut)}
        elif kind == 'partition':
            a = float(x @ np.array(d['durations']))
            b = float(sum(d['durations']) - a)
            metrics = {'load_a': a, 'load_b': b, 'load_difference': abs(a-b)}
            if self.spec.variant=='coupled':
                penalty=d['balance_coefficient']*(a-b)**2
                metrics.update(contention_cost=self.energy(x)-penalty,balance_cost=penalty)
        elif kind == 'portfolio':
            k = d['choose']
            if sum(x) != k:
                violations.append('cardinality')
            weights = x / k  # intended equal weights; invalid count remains visible
            metrics = {'selected_assets': int(sum(x)), 'required_assets': k,
                       'expected_return_percent': float(weights @ np.array(d['returns'])*100),
                       'volatility_percent': float(np.sqrt(weights @ np.array(d['covariance']) @ weights)*100)}
        elif kind == 'projects':
            conflict_count = sum(int(x[i]*x[j]) for i,j in d.get('conflicts',[]))
            if conflict_count:
                violations.append('conflicts')
            metrics = {'benefit': float(x @ np.array(d['benefits'])), 'conflicts': conflict_count}
            if self.spec.variant=='interactions':
                metrics.pop('conflicts')
                metrics.update(pair_cost=self.energy(x)+metrics['benefit'],net_value=-self.energy(x),selected_count=int(x.sum()))
        elif kind == 'assignment':
            if self.spec.variant=='two_pools':
                costs=np.asarray(d['costs'])
                execution=float(costs[np.arange(self.n),x].sum())
                metrics={'execution_cost':execution,'interaction_cost':self.energy(x)-execution,
                         'pool_a_count':len(other),'pool_b_count':len(chosen)}
                return {'selected':other,'other_group':chosen,'metrics':metrics,'feasible':True,'violations':[]}
            n = self.spec.size
            grid = x.reshape(n,n)
            bad = int(np.sum(grid.sum(axis=0) != 1) + np.sum(grid.sum(axis=1) != 1))
            if bad:
                violations.append('assignment')
            metrics = {'cost': float(np.sum(grid * np.array(d['costs']))), 'invalid_rows_columns': bad}
            chosen = [{'worker': int(i)+1, 'task': int(j)+1} for i,j in zip(*np.nonzero(grid))]
            other = []
        return {'selected': chosen, 'other_group': other if kind in ('channels','maxcut','partition') else [],
                'metrics': metrics, 'feasible': not violations, 'violations': violations}


def build_problem(spec: Spec) -> Problem:
    if spec.variant!='standard':
        from .large_problems import build_large
        return build_large(spec)
    rng = np.random.default_rng(spec.seed)
    n, kind = spec.size, spec.kind
    q, offset, data, weights = np.zeros((n,n), dtype=float), 0.0, {}, None
    if kind == 'channels':
        # Exactly the same generator and random draw shape as the benchmark baseline.
        upper = np.triu(rng.integers(1, 11, size=(n, n), dtype=np.int16), 1)
        weights = np.asarray(spec.weights if spec.weights is not None else upper + upper.T, dtype=np.float64)
        q = weights.copy()
        q[np.diag_indices(n)] = -weights.sum(axis=1)
        offset = float(weights.sum() / 2)
        data = {'channels': 2, 'edges': n*(n-1)//2, 'total_interference_weight': offset,
                'generator': 'numpy.default_rng(seed); U=triu(rng.integers(1,11,(n,n),dtype=int16),1); W=(U+U.T).astype(float64)',
                'weight_range': [float(weights.min()), float(weights.max())], 'model': 'Two-session audience conflicts. No capacities or time-window constraints.'}
        if spec.user_data(): data.pop('generator', None)
    elif kind == 'maxcut':
        edges = []
        density = .35 if n <= 32 else .12
        for i in range(n):
            for j in range(i+1,n):
                if rng.random() < density:
                    w = int(rng.integers(1,11))
                    edges.append([i,j,w])
                    q[i,i] -= w; q[j,j] -= w
                    q[i,j] += w; q[j,i] += w
        data = {'edges': edges}
    elif kind == 'partition':
        durations = np.asarray(spec.durations,dtype=float) if spec.durations is not None else rng.integers(1,31,size=n)
        total = float(sum(durations))
        q = 4 * np.outer(durations, durations).astype(float)
        q[np.diag_indices(n)] -= 4 * total * durations
        offset = float(total**2)
        data = {'durations': durations.tolist()}
    elif kind == 'portfolio':
        k = spec.choose or max(1, n//4)
        factors = rng.normal(0,.04,size=(n,5))
        cov = factors @ factors.T + np.diag(rng.uniform(.003,.012,n))
        returns = rng.uniform(.025,.16,n)
        base = 5*cov/k**2
        base[np.diag_indices(n)] -= returns/k
        penalty = float(np.abs(base).sum()+1)
        q = base + penalty*np.ones((n,n))
        q[np.diag_indices(n)] -= 2*penalty*k
        offset = penalty*k*k
        data = {'choose': k, 'returns': returns.tolist(), 'covariance': cov.tolist(),
                'risk_coefficient': 5, 'penalty': penalty}
    elif kind == 'projects':
        benefits = np.asarray(spec.benefits,dtype=float) if spec.benefits is not None else rng.integers(1,21,n)
        penalty = float(max(benefits))+1
        q = -np.diag(benefits).astype(float)
        conflicts = []
        for i in range(n):
            for j in range(i+1,n):
                if (([i+1,j+1] in spec.conflicts or [j+1,i+1] in spec.conflicts) if spec.conflicts is not None else rng.random() < .08):
                    conflicts.append([i,j]); q[i,j] = q[j,i] = penalty/2
        data = {'benefits': benefits.tolist(), 'conflicts': conflicts, 'penalty': penalty}
    elif kind == 'assignment':
        costs = np.asarray(spec.costs,dtype=float) if spec.costs is not None else rng.integers(1,81,(n,n))
        penalty = float(n*costs.max()+1)
        q = np.diag(costs.ravel()).astype(float)
        for ids in ([i*n+j for j in range(n)] for i in range(n)):
            q[np.ix_(ids,ids)] += penalty
            q[ids,ids] -= 2*penalty
        for ids in ([i*n+j for i in range(n)] for j in range(n)):
            q[np.ix_(ids,ids)] += penalty
            q[ids,ids] -= 2*penalty
        offset = float(2*n*penalty)
        data = {'costs': costs.tolist(), 'penalty': penalty}
    elif kind == 'custom':
        raw = np.array(spec.matrix,dtype=float)
        q = (raw+raw.T)/2  # preserves x.T Q x, also supports triangular input
        offset = float(spec.offset)
    return Problem(spec, q, offset, data, weights)


def exact_reference(p: Problem):
    """Independent verifier, never feeds either heuristic. No exponential search above 18 bits."""
    if p.spec.kind == 'assignment' and p.spec.variant=='standard':
        n = p.spec.size
        costs = np.array(p.data['costs'])
        dp = {0: (0, [])}
        for i in range(n):
            nxt = {}
            for mask, (value, path) in dp.items():
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    key = mask | (1 << j)
                    candidate = value + float(costs[i,j])
                    if key not in nxt or candidate < nxt[key][0]:
                        nxt[key] = (candidate, path+[j])
            dp = nxt
        value, path = dp[(1 << n)-1]
        x = np.zeros(p.n, dtype=int)
        for i,j in enumerate(path):
            x[i*n+j]=1
        return {'known': True, 'energy': float(value), 'bits': x.tolist(),
                'method': 'exact_assignment_dynamic_programming',
                'proof': 'Nonnegative costs; penalty exceeds the cost of every feasible assignment.'}
    if p.n > 18:
        return {'known': False, 'energy': None, 'method': 'not_computed', 'bits': None}
    best, bits = float('inf'), None
    for start in range(0, 1 << p.n, 2048):
        ids = np.arange(start,min(start+2048,1 << p.n),dtype=np.uint32)
        states = ((ids[:,None] >> np.arange(p.n,dtype=np.uint32)) & 1).astype(float)
        energies = np.sum((states @ p.q)*states,axis=1)+p.offset
        i = int(np.argmin(energies))
        if energies[i] < best:
            best, bits = float(energies[i]), states[i].astype(int).tolist()
    return {'known': True, 'energy': best, 'bits': bits, 'method': 'complete_enumeration'}


def preset_for_text(text):
    stripped = text.strip()
    return next((p for p in PRESETS if stripped in p['prompts'].values()), None)
