"""Private on-demand NVIDIA worker. Deploy with: python -m modal deploy modal_app.py"""
import os
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parent
app=modal.App('metaquant-forge')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.6.0',index_url='https://download.pytorch.org/whl/cu124')
       .pip_install('numpy==2.5.4','pydantic==2.14.1','dimod==0.12.22',
                    'dwave-samplers==1.8.0','simulated-bifurcation==2.0.0')
       .env({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
       .add_local_dir(ROOT/'optimizer',remote_path='/root/optimizer'))
updates=modal.Dict.from_name('metaquant-forge-progress',create_if_missing=True)

@app.function(image=image,gpu='L4',cpu=4,memory=16384,timeout=600,
              scaledown_window=20,min_containers=0,max_containers=1,retries=0)
def compare_job(job_id:str,spec_data:dict,seed:int):
    import sys,time
    sys.path.insert(0,'/root')
    from optimizer.specs import Spec
    from optimizer.problems import build_problem
    from optimizer.engines import verify_gpu,compare
    from optimizer.live import prepare
    if len(job_id)!=32 or any(c not in '0123456789abcdef' for c in job_id):raise ValueError('Invalid job identity')
    def stage(name):updates[job_id]={'stage':name}
    stage('warming_up')
    hw=verify_gpu()
    prepare()
    p=build_problem(Spec.model_validate(spec_data))
    stage('running')
    def partial(result):updates[job_id]={'stage':'running','partial':result}
    result=compare(p,seed=seed,progress=stage,partial=partial)
    result['provider']='Modal'
    updates[job_id]={'stage':'completed','partial':result}
    return result
