"""Small Render gateway; GPU and CPU solver bundles run privately on Modal."""
import asyncio,hashlib,hmac,json,logging,os,secrets,threading,time,traceback
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID,uuid4
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import JSONResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field,ConfigDict
from typing import Literal
from optimizer.catalog import CATEGORIES,PRESETS
from optimizer.specs import Spec,resolve_input,InputError
from . import meta_ai

log=logging.getLogger('metaquant')
log.setLevel(logging.INFO)
if not log.hasHandlers():
    handler=logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log.addHandler(handler)
ROOT=Path(__file__).resolve().parents[1]
SESSION_KEY=os.getenv('SESSION_SECRET','') or secrets.token_hex(32)
JOBS={}; LOCK=threading.RLock(); RECENT={}; ACTIVE=None; PLANNING=threading.Lock()
TERMINAL={'completed','failed','cancelled'}
MAX_SECONDS=720
LISTENERS={}
PROGRESS_INTERVAL=0.25

def _signal(queue):
    if queue.empty():queue.put_nowait(True)

def update_job(key,**changes):
    """Publish state changes to browser streams; log phases without user data."""
    with LOCK:
        job=JOBS[key]
        started=job.setdefault('_started',time.monotonic())
        elapsed=round(1000*(time.monotonic()-started),2)
        stage=changes.get('stage')
        if stage and stage!=job.get('stage'):
            history=job.setdefault('gateway_timeline',[])
            history.append({'stage':stage,'gateway_ms':elapsed})
            log.info('Job %s stage=%s gateway_ms=%.2f',key,stage,elapsed)
        previous=set(job.get('partial',{}).get('engines',{}))
        incoming=changes.get('partial',{}).get('engines',{})
        for name in set(incoming)-previous:
            log.info('Job %s result=%s received_ms=%.2f bundle_wall_ms=%s',
                     key,name,elapsed,incoming[name].get('bundle_wall_ms'))
        job.update(changes)
        job['gateway_elapsed_ms']=elapsed
        for loop,queue in tuple(LISTENERS.get(key,())):
            if not loop.is_closed():loop.call_soon_threadsafe(_signal,queue)

class TextRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    question:str=Field(min_length=1,max_length=100000)
    language:Literal['en','es','fr','de']='en'
class RunRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request_id:UUID
    spec:Spec
    language:Literal['en','es','fr','de']='en'

def gpu_configured():
    return bool(os.getenv('MODAL_TOKEN_ID') and os.getenv('MODAL_TOKEN_SECRET'))

def modal_environment():
    return os.getenv('MODAL_ENVIRONMENT','').strip() or 'main'

def log_job_failure(key,phase,exc):
    # Keep useful tracebacks in Render logs, never credentials or public API responses.
    detail=''.join(traceback.format_exception(exc))
    for name in ('MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','META_API_KEY','SESSION_SECRET'):
        value=os.getenv(name)
        if value:detail=detail.replace(value,'[REDACTED]')
    log.error('Job %s failed during %s (environment=%s):\n%s',key,phase,modal_environment(),detail)

def sign(value):return hmac.new(SESSION_KEY.encode(),value.encode(),hashlib.sha256).hexdigest()
def owner(request):return request.state.owner

def rate(key,limit=12):
    now=time.monotonic()
    with LOCK:
        # Bounded map; a new visitor cannot grow process memory indefinitely.
        if key not in RECENT and len(RECENT)>2048:RECENT.pop(next(iter(RECENT)))
        q=RECENT.setdefault(key,deque())
        while q and now-q[0]>60:q.popleft()
        if len(q)>=limit:raise HTTPException(429,'rate_limit')
        q.append(now)

def snapshot(job):return {k:v for k,v in job.items() if not k.startswith('_')}
def lookup(request,key):
    job=JOBS.get(key)
    if not job or job['_owner']!=owner(request):raise HTTPException(404,'job_expired')
    return job

def valid_result(r):
    try:
        hw=r['hardware'];q=r['engines']['quantum']
        return (r['schema_version']=='metaquant-forge-v1' and r['complete'] is True
            and hw['sb_gpu_vendor']=='NVIDIA CUDA' and hw['gpu_execution_verified'] is True
            and hw['cpu_fallback'] is False and bool(hw['cuda_version'])
            and set(r['engines'])=={'traditional','quantum'}
            and all(len(e['trials'])==3 for e in r['engines'].values())
            and all(x.get('cpu_fallback') is False and all(x.get(k)=='cuda:0' for k in ('matrix_device','solution_device','objective_device')) for x in [q,*q['trials']]))
    except (KeyError,TypeError):return False

def run_job(key):
    global ACTIVE
    call=None;updates=None;phase='load_sdk';progress_available=True
    try:
        import modal
        job=JOBS[key]
        update_job(key,stage='connecting')
        environment=modal_environment()
        phase='lookup_function'
        fn=modal.Function.from_name('metaquant-forge','compare_job',environment_name=environment)
        fn.hydrate()
        phase='initialize_progress'
        # from_name is lazy. Create the shared Dict before the cold worker starts;
        # otherwise the first progress read can raise NotFoundError and cancel it.
        updates=modal.Dict.from_name('metaquant-forge-progress',environment_name=environment,create_if_missing=True)
        updates.hydrate()
        seed=1000+(job['spec']['seed']%100000)*100
        phase='start_function'
        call=fn.spawn(key,job['spec'],seed)
        with LOCK:job['_call']=call
        update_job(key,stage='waiting_worker')
        start=time.monotonic()
        last_packet=None
        while True:
            with LOCK:cancel=job.get('_cancel',False)
            if cancel or time.monotonic()-start>MAX_SECONDS:
                call.cancel(terminate_containers=True)
                update_job(key,status='cancelled' if cancel else 'failed',stage='cancelled' if cancel else 'job_timeout',error='cancelled' if cancel else 'job_timeout')
                break
            packet=None
            if progress_available:
                try:packet=updates.get(key,None)
                except Exception as exc:
                    # Progress is advisory: the FunctionCall is the source of truth.
                    log_job_failure(key,'read_progress (continuing computation)',exc)
                    progress_available=False
            if packet and packet!=last_packet:
                last_packet=packet
                changes={'stage':packet.get('stage','running')}
                # "Completed" is only shown after validating the final result.
                if changes['stage']=='completed':changes['stage']='receiving_result'
                if packet.get('partial'):changes['partial']=packet['partial']
                if packet.get('worker_timeline'):changes['worker_timeline']=packet['worker_timeline']
                update_job(key,**changes)
            try:
                phase='read_result'
                result=call.get(timeout=0)
                if not valid_result(result):raise RuntimeError('gpu_verification_failed')
                update_job(key,status='completed',stage='completed',result=result)
                break
            except modal.exception.FunctionTimeoutError:
                update_job(key,status='failed',stage='job_timeout',error='job_timeout')
                break
            except (TimeoutError,modal.exception.TimeoutError):pass
            time.sleep(PROGRESS_INTERVAL)
    except Exception as exc:
        log_job_failure(key,phase,exc)
        code='compute_failed'
        if type(exc).__name__=='NotFoundError':
            code='modal_app_not_found' if phase in {'lookup_function','start_function'} else 'modal_resource_not_found'
        elif type(exc).__name__ in {'AuthError','PermissionDeniedError'}:code='modal_access_denied'
        elif type(exc).__name__=='DeserializationError':code='result_transfer_failed'
        if call:
            try:call.cancel(terminate_containers=True)
            except Exception:pass
        update_job(key,status='failed',stage=code,error=code)
    finally:
        if updates is not None and call is not None:
            try:updates.pop(key,None)
            except Exception:pass
        with LOCK:
            JOBS[key].pop('_call',None)
            if ACTIVE==key:ACTIVE=None

@asynccontextmanager
async def lifespan(app):
    yield
    with LOCK:
        for job in JOBS.values():
            if job['status'] not in TERMINAL:job['_cancel']=True

app=FastAPI(title='MetaQuant Forge',lifespan=lifespan,docs_url=None,redoc_url=None)
@app.middleware('http')
async def session(request:Request,call_next):
    raw=request.cookies.get('mq_session','');parts=raw.split('.')
    valid=len(parts)==2 and len(parts[0])==32 and hmac.compare_digest(sign(parts[0]),parts[1])
    sid=parts[0] if valid else uuid4().hex
    request.state.owner=sid
    if request.method=='POST':
        if request.headers.get('x-metaquant-request')!='1':return JSONResponse({'detail':'invalid_request'},403)
        try:
            if int(request.headers.get('content-length','0'))>150000:return JSONResponse({'detail':'input_too_large'},413)
        except ValueError:return JSONResponse({'detail':'invalid_request'},400)
        if request.headers.get('content-type','').split(';')[0]!='application/json':return JSONResponse({'detail':'invalid_request'},415)
    response=await call_next(request)
    if not valid:response.set_cookie('mq_session',sid+'.'+sign(sid),httponly=True,samesite='strict',secure=os.getenv('COOKIE_SECURE','1')=='1',max_age=86400)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='same-origin'
    response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    if request.url.path.startswith('/api/'):response.headers['Cache-Control']='no-store'
    return response

@app.get('/health')
def health():return {'status':'ok','version':'1.0.0'}
@app.get('/api/info')
def info():
    return {'version':'1.0.0','gpu_configured':gpu_configured(),'meta_configured':meta_ai.configured(),
        'categories':CATEGORIES,'presets':PRESETS,'model':meta_ai.configuration()[1],
        'busy':ACTIVE is not None,'streaming':True,'delivery_version':'2026-10-22-live-results',
        'storage':'transient; download results before restart'}
@app.post('/api/plan')
def plan(body:TextRequest,request:Request):
    rate('ai-global',10);rate(owner(request)+'-ai',5)
    if not PLANNING.acquire(blocking=False):raise HTTPException(429,'busy')
    try:return meta_ai.formulate(body.question,body.language)
    except meta_ai.AIError as e:raise HTTPException(422,str(e)) from None
    finally:PLANNING.release()
@app.post('/api/validate')
def validate(body:TextRequest):
    try:
        spec,adapter=resolve_input(body.question)
        return {'spec':spec.model_dump(exclude_none=True),'source':adapter['source']}
    except InputError as e:raise HTTPException(422,{'code':str(e),'issues':e.issues}) from None
@app.post('/api/jobs')
def submit(body:RunRequest,request:Request):
    global ACTIVE
    if not gpu_configured():raise HTTPException(503,'gpu_not_configured')
    key=body.request_id.hex;spec=body.spec.model_dump(exclude_none=True)
    with LOCK:
        if key in JOBS:
            job=lookup(request,key)
            if job['spec']!=spec:raise HTTPException(409,'request_id_conflict')
            return snapshot(job)
        if ACTIVE:raise HTTPException(429,'busy')
        rate('gpu-global',20)
        if len(JOBS)>=20:
            old=next((k for k,v in JOBS.items() if v['status'] in TERMINAL),None)
            if old:JOBS.pop(old)
        job={'id':key,'spec':spec,'status':'running','stage':'connecting','created_at':time.time(),
             '_owner':owner(request),'_language':body.language,'_cancel':False,
             '_started':time.monotonic(),'gateway_timeline':[{'stage':'connecting','gateway_ms':0}]}
        JOBS[key]=job;ACTIVE=key
        threading.Thread(target=run_job,args=(key,),daemon=True).start()
        return snapshot(job)
@app.get('/api/jobs/{key}')
def get_job(key:str,request:Request):
    with LOCK:return snapshot(lookup(request,key))

async def stream_job(key,request):
    loop=asyncio.get_running_loop();queue=asyncio.Queue(maxsize=1)
    listener=(loop,queue)
    with LOCK:LISTENERS.setdefault(key,set()).add(listener)
    try:
        while True:
            if await request.is_disconnected():return
            with LOCK:
                job=JOBS.get(key)
                if job is None or job.get('_owner')!=owner(request):
                    payload={'status':'failed','stage':'job_expired','error':'job_expired'}
                else:payload=snapshot(job)
            yield 'data: '+json.dumps(payload,allow_nan=False,separators=(',',':'))+'\n\n'
            if payload['status'] in TERMINAL:return
            while True:
                try:
                    await asyncio.wait_for(queue.get(),timeout=15)
                    break
                except asyncio.TimeoutError:
                    if await request.is_disconnected():return
                    yield 'event: heartbeat\ndata: {}\n\n'
    finally:
        with LOCK:
            listeners=LISTENERS.get(key,set());listeners.discard(listener)
            if not listeners:LISTENERS.pop(key,None)

@app.get('/api/jobs/{key}/events')
async def events(key:str,request:Request):
    with LOCK:lookup(request,key)
    return StreamingResponse(stream_job(key,request),media_type='text/event-stream',
        headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})
@app.post('/api/jobs/{key}/cancel')
def cancel(key:str,request:Request):
    with LOCK:
        job=lookup(request,key)
        if job['status'] not in TERMINAL:update_job(key,_cancel=True,stage='cancelling')
        return snapshot(job)
@app.post('/api/jobs/{key}/explain')
def explain(key:str,request:Request):
    with LOCK:
        job=lookup(request,key)
        if job['status']!='completed':raise HTTPException(409,'not_completed')
        if 'explanation' in job:return job['explanation']
        result=job['result'];language=job['_language']
    rate('ai-global',10);rate(owner(request)+'-ai',5)
    if not PLANNING.acquire(blocking=False):raise HTTPException(429,'busy')
    try:
        value=meta_ai.explain(result,language)
        with LOCK:job['explanation']=value
        return value
    except meta_ai.AIError as e:raise HTTPException(422,str(e)) from None
    finally:PLANNING.release()

app.mount('/',StaticFiles(directory=ROOT/'dist',html=True),name='web')
