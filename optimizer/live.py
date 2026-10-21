"""Warm isolated solver processes, real completion order, no artificial delays."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import time
import numpy as np

_pools={}
_cancel=None

def initialize(name, cancel):
    global _cancel
    _cancel=cancel
    from .engines import verify_gpu, classical
    from .problems import Spec, build_problem
    if name=='quantum': verify_gpu()
    else: classical(build_problem(Spec(kind='channels',size=4,seed=3)),3)

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

def bundle(name,p,seed):
    from .engines import classical, bifurcation, REPEATS
    rows=[]
    started=time.perf_counter()
    for i in range(REPEATS):
        if _cancel is not None and _cancel.is_set(): raise RuntimeError('cancelled')
        row=(classical if name=='traditional' else bifurcation)(p,seed+i)
        checked=p.energy(row['bits'])
        if not np.isclose(checked,row['energy'],rtol=1e-9,atol=1e-6):
            raise RuntimeError('objective_verification_failed')
        row['independent_cpu_energy']=checked
        rows.append(row)
    best=min(rows,key=lambda row:row['energy'])
    return {**best,'decoded':p.decode(best['bits']),'trials':rows,
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
