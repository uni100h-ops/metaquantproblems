"""Warm isolated solver processes, real completion order, no artificial delays."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import time
import numpy as np

_pools={}
_cancel=None
CPU_TRIAL_WORKERS=3
REPEATS=3

def initialize(name, cancel):
    global _cancel
    _cancel=cancel
    from .problems import Spec, build_problem
    if name=='quantum':
        from .engines import verify_gpu
        verify_gpu()
    else:
        from .cpu_engines import classical
        classical(build_problem(Spec(kind='channels',size=4,seed=3)),3)

def ready():
    return True

def prepare():
    if _pools: return
    context=mp.get_context('spawn')
    try:
        for name in ('traditional','quantum'):
            event=context.Event()
            executor=ProcessPoolExecutor(max_workers=1,mp_context=context,
                initializer=initialize,initargs=(name,event))
            _pools[name]=(executor,event)
        ready_futures=[executor.submit(ready) for executor,event in _pools.values()]
        for future in ready_futures:future.result(timeout=180)
    except BaseException:
        close()
        raise

def close():
    for executor,event in _pools.values():
        event.set()
        executor.shutdown(wait=True,cancel_futures=True)
    _pools.clear()

def collect_trials(solver,p,seed,repetitions,workers=1):
    """Independent seeds; native dwave-samplers 1.8.0 releases the GIL.

    Its RNG is thread-local. Each call owns its prepared model and initial
    states; only the read-only Problem is shared. Keep rows in seed order.
    GPU trials remain sequential because they share the CUDA RNG/context.
    """
    def trial(index):
        if _cancel is not None and _cancel.is_set(): raise RuntimeError('cancelled')
        row=solver(p,seed+index)
        if _cancel is not None and _cancel.is_set(): raise RuntimeError('cancelled')
        checked=p.energy(row['bits'])
        if not np.isclose(checked,row['energy'],rtol=1e-9,atol=1e-6):
            raise RuntimeError('objective_verification_failed')
        row['independent_cpu_energy']=checked
        return row
    if workers==1:
        return [trial(i) for i in range(repetitions)]
    with ThreadPoolExecutor(max_workers=min(workers,repetitions),
                            thread_name_prefix='cpu-trial') as executor:
        return list(executor.map(trial,range(repetitions)))

def bundle(name,p,seed):
    if name=='traditional':
        from .cpu_engines import classical as solver
    else:
        from .engines import bifurcation as solver
    started=time.perf_counter()
    workers=min(CPU_TRIAL_WORKERS,REPEATS) if name=='traditional' else 1
    rows=collect_trials(solver,p,seed,REPEATS,workers)
    best=min(rows,key=lambda row:row['energy'])
    return {**best,'decoded':p.decode(best['bits']),'trials':rows,
            'trial_workers':workers,
            'trial_execution':'parallel_threads' if workers>1 else 'sequential',
            'solver_ms':sum(row['elapsed_ms'] for row in rows),
            'median_ms':float(np.median([row['elapsed_ms'] for row in rows])),
            'optimality_gap_absolute':None,'optimality_gap_percent':None,
            'at_proven_optimum':False,'bundle_wall_ms':1000*(time.perf_counter()-started)}

def run_bundles(p,seed,on_result,cancelled):
    prepare()
    for executor,event in _pools.values(): event.clear()
    started=time.perf_counter()
    futures={executor.submit(bundle,name,p,seed):name for name,(executor,event) in _pools.items()}
    errors=[]
    while futures:
        if cancelled():
            for executor,event in _pools.values(): event.set()
        finished,_=wait(futures,timeout=.1,return_when=FIRST_COMPLETED)
        for future in finished:
            name=futures.pop(future)
            try:
                engine=future.result()
                engine['ready_after_ms']=1000*(time.perf_counter()-started)
                if not cancelled(): on_result(name,engine)
            except Exception as exc:
                errors.append(exc)
        # Let the other engine finish; preserve any independently valid result.
    if cancelled(): raise RuntimeError('cancelled')
    if errors: raise errors[0]
