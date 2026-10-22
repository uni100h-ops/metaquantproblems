"""Real CPU solver equivalence plus scheduling, cancellation and timing contracts."""
from concurrent.futures import Future
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from optimizer import live
from optimizer.problems import Spec, build_problem
from optimizer.channel_engines import CPU_PROFILE, run_cpu
from optimizer.dense_engines import classical_dense


def native_cpu(p, seed):
    if p.spec.kind != 'channels':
        return classical_dense(p, seed)
    raw = run_cpu(p.weights, CPU_PROFILE, seed, None, 1)
    bits = ((np.asarray(raw['spins']) + 1) // 2).tolist()
    assert p.energy(bits) == pytest.approx(raw['remaining_interference'])
    return dict(bits=bits, energy=raw['remaining_interference'], seed=seed,
                elapsed_ms=raw['total_ms'])


@pytest.fixture(autouse=True)
def clean_cancel(monkeypatch):
    monkeypatch.setattr(live, '_cancel', None)


@pytest.mark.parametrize('kind,variant', [
    ('channels','standard'), ('partition','coupled'),
    ('projects','interactions'), ('assignment','two_pools'), ('custom','dense')])
def test_native_parallel_matches_sequential(kind, variant):
    p = build_problem(Spec(kind=kind, variant=variant, size=64, seed=20261026))
    before = p.q.copy()
    sequential = live.collect_trials(native_cpu, p, 6103600, 3, workers=1)
    parallel = live.collect_trials(native_cpu, p, 6103600, 3, workers=3)
    assert [r['seed'] for r in parallel] == [6103600,6103601,6103602]
    for a,b in zip(sequential,parallel):
        assert a['bits'] == b['bits']
        assert a['energy'] == b['energy'] == b['independent_cpu_energy']
    np.testing.assert_array_equal(before,p.q)


def test_trials_really_overlap_and_preserve_seed_order():
    p = build_problem(Spec(kind='channels',size=4,seed=3))
    barrier = threading.Barrier(3, timeout=5)
    thread_ids = set()
    lock = threading.Lock()
    def solver(problem,seed):
        with lock:
            thread_ids.add(threading.get_ident())
        barrier.wait()  # A sequential implementation cannot pass this barrier.
        return native_cpu(problem,seed)
    rows = live.collect_trials(solver,p,10,3,workers=3)
    assert len(thread_ids) == 3
    assert [r['seed'] for r in rows] == [10,11,12]


def test_cancelled_trials_do_not_start(monkeypatch):
    event = threading.Event(); event.set()
    monkeypatch.setattr(live,'_cancel',event)
    def forbidden(*args):
        raise AssertionError('solver must not start')
    with pytest.raises(RuntimeError,match='cancelled'):
        live.collect_trials(forbidden,None,10,3,workers=3)


def test_cancelled_result_is_not_published(monkeypatch):
    event = threading.Event()
    monkeypatch.setattr(live,'_cancel',event)
    p = build_problem(Spec(kind='channels',size=4,seed=3))
    def solver(problem,seed):
        row = native_cpu(problem,seed)
        event.set()
        return row
    with pytest.raises(RuntimeError,match='cancelled'):
        live.collect_trials(solver,p,10,3,workers=3)


def test_invalid_energy_fails_verification():
    p = build_problem(Spec(kind='channels',size=4,seed=3))
    def solver(problem,seed):
        return dict(bits=[0]*4,energy=-999,elapsed_ms=1)
    with pytest.raises(RuntimeError,match='objective_verification_failed'):
        live.collect_trials(solver,p,10,3,workers=3)


def test_bundle_gpu_stays_sequential_and_cpu_uses_three_threads(monkeypatch):
    p = build_problem(Spec(kind='channels',size=4,seed=3))
    calls = []
    fake = SimpleNamespace(classical=object(),bifurcation=object(),REPEATS=3)
    monkeypatch.setitem(sys.modules,'optimizer.engines',fake)
    rows = [dict(bits=[0]*4,energy=p.energy([0]*4),elapsed_ms=t,seed=10+i)
            for i,t in enumerate([11,12,13])]
    def collect(solver,problem,seed,repetitions,workers):
        calls.append((solver,repetitions,workers))
        return rows
    monkeypatch.setattr(live,'collect_trials',collect)
    for name,workers in [('traditional',3),('quantum',1)]:
        result = live.bundle(name,p,10)
        assert result['trial_workers'] == workers
        assert result['solver_ms'] == 36
        assert result['median_ms'] == 12
        assert result['bundle_wall_ms'] >= 0
        assert result['seed'] == 10  # Stable tie breaking in seed order.
    assert calls == [(fake.classical,3,3),(fake.bifurcation,3,1)]


def test_fast_engine_is_emitted_before_slow_engine(monkeypatch):
    slow,fast = Future(),Future()
    fast.set_result({'engine':'GPU'})
    event = threading.Event()
    monkeypatch.setattr(live,'prepare',lambda:None)
    monkeypatch.setattr(live,'_pools',{
        'traditional':(SimpleNamespace(submit=lambda *a:slow),event),
        'quantum':(SimpleNamespace(submit=lambda *a:fast),event)})
    completed = []
    def on_result(name,engine):
        completed.append(name)
        assert engine['ready_after_ms'] >= 0
        if name == 'quantum':
            assert not slow.done()
            slow.set_result({'engine':'CPU'})
    live.run_bundles(None,10,on_result,lambda:False)
    assert completed == ['quantum','traditional']


def test_comparison_uses_elapsed_time_and_exports_the_original_sum(monkeypatch):
    # Load the production orchestrator without a CUDA dependency. Solver/GPU
    # calls are replaced; this tests timing semantics, not GPU execution.
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(set_num_threads=lambda n:None))
    monkeypatch.setitem(sys.modules,'simulated_bifurcation',SimpleNamespace())
    spec = importlib.util.spec_from_file_location('optimizer._timing_test',
        Path(live.__file__).with_name('engines.py'))
    engines = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engines)
    monkeypatch.setattr(engines,'device_name',lambda:'cuda:0')
    monkeypatch.setattr(engines,'hardware',lambda:{'test_fixture':True})
    monkeypatch.setattr(engines,'_gpu_verified',True)
    monkeypatch.setattr(engines,'_warmed',True)
    p = build_problem(Spec(kind='channels',size=4,seed=3))
    row = dict(bits=[0]*4,energy=p.energy([0]*4),decoded=p.decode([0]*4))
    def run(problem,seed,on_result,cancelled):
        on_result('quantum',dict(row,solver_ms=1000,bundle_wall_ms=1100))
        on_result('traditional',dict(row,solver_ms=15000,bundle_wall_ms=5500))
    monkeypatch.setattr(live,'run_bundles',run)
    result = engines.compare(p,seed=10)
    assert result['comparison']['sb_speed_ratio'] == 5
    assert result['comparison']['sb_summed_call_ratio'] == 15
    assert result['comparison']['speed_ratio_basis'] == 'bundle_wall_ms'
    protocol = result['protocol']
    assert protocol['repetitions'] == 3
    assert protocol['traditional_trial_workers'] == 3
    assert protocol['traditional_workers_per_trial'] == protocol['sb_trial_workers'] == 1
    assert protocol['traditional_candidates'] == 8 and protocol['traditional_sweeps'] == 200
    assert protocol['sb_candidates'] == 128 and protocol['sb_max_steps'] == 1000


def test_ai_explanation_receives_elapsed_and_summed_timings(monkeypatch):
    from server import meta_ai
    seen = {}
    def chat(system,user):
        seen.update(system=system,user=user)
        return 'test explanation',{}
    monkeypatch.setattr(meta_ai,'chat',chat)
    meta_ai.explain(dict(problem={},reference={},protocol={},engines={
        'traditional':dict(energy=1,solver_ms=15000,median_ms=5000,
            bundle_wall_ms=5500,trial_workers=3,trial_execution='parallel_threads',
            decoded={'selected':[1,2],'metrics':{}},at_proven_optimum=False)}),'English')
    import json
    engine = json.loads(seen['user'])['engines']['traditional']
    assert engine['bundle_wall_ms'] == 5500 and engine['solver_ms'] == 15000
    assert engine['trial_workers'] == 3
    assert 'selected' not in engine['decoded']
    assert 'can exceed elapsed time' in seen['system']
