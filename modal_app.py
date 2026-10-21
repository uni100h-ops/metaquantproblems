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

# Import the CUDA library during container initialization, before Modal starts
# handling inputs. It is deliberately not required on the deployment computer.
# Do not initialize CUDA here; worker verification below still runs on the GPU.
with image.imports():
    import torch

updates=modal.Dict.from_name('metaquant-forge-progress',create_if_missing=True)
_worker_ready=False

# Fall back to T4 capacity when L4 workers are unavailable. Both run the same
# CUDA solvers; hardware() reports the actual GPU, with no CPU substitution.
@app.function(image=image,gpu=['L4','T4'],cpu=4,memory=16384,timeout=600,
              scaledown_window=300,min_containers=0,max_containers=1,retries=0)
def compare_job(job_id:str,spec_data:dict,seed:int):
    global _worker_ready
    import sys
    sys.path.insert(0,'/root')
    from optimizer.wire import ProgressPublisher
    if len(job_id)!=32 or any(c not in '0123456789abcdef' for c in job_id):raise ValueError('Invalid job identity')
    publisher=ProgressPublisher(lambda packet:updates.put(job_id,packet))
    publisher.stage('initializing' if not _worker_ready else 'preparing_problem')
    from optimizer.specs import Spec
    from optimizer.problems import build_problem
    from optimizer.engines import verify_gpu,compare
    from optimizer.live import prepare
    if not _worker_ready:
        publisher.stage('warming_up')
        verify_gpu()
        prepare()
        _worker_ready=True
    publisher.stage('preparing_problem')
    p=build_problem(Spec.model_validate(spec_data))
    publisher.stage('running')
    result=compare(p,seed=seed,progress=publisher.stage,partial=publisher.partial)
    result['provider']='Modal'
    return publisher.complete(result)
