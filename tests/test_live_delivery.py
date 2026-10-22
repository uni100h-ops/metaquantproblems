"""Actual stream framing, ownership, phase telemetry and CPU-only startup."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4
from fastapi.testclient import TestClient
import pytest
from server import main
from optimizer.wire import ProgressPublisher


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    main.JOBS.clear();main.LISTENERS.clear();main.ACTIVE=None
    monkeypatch.setenv('COOKIE_SECURE','0')
    yield
    main.JOBS.clear();main.LISTENERS.clear();main.ACTIVE=None


def test_stream_delivers_gpu_without_waiting_for_cpu_and_closes():
    async def check():
        key='a'*32
        async def connected():return False
        request=SimpleNamespace(state=SimpleNamespace(owner='owner'),is_disconnected=connected)
        main.JOBS[key]={'id':key,'_owner':'owner','status':'running','stage':'waiting_worker'}
        stream=main.stream_job(key,request)
        first=await anext(stream)
        assert json.loads(first.removeprefix('data: '))['stage']=='waiting_worker'
        waiting=asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        assert not waiting.done()
        partial={'engines':{'quantum':{'bundle_wall_ms':783.4}},'complete':False}
        main.update_job(key,stage='waiting_cpu',partial=partial)
        packet=json.loads((await asyncio.wait_for(waiting,1)).removeprefix('data: '))
        assert packet['status']=='running' and set(packet['partial']['engines'])=={'quantum'}
        main.update_job(key,status='completed',stage='completed',result={'complete':True})
        final=json.loads((await anext(stream)).removeprefix('data: '))
        assert final['status']=='completed'
        with pytest.raises(StopAsyncIteration):await anext(stream)
        assert not main.LISTENERS
    asyncio.run(check())


def test_stream_disconnect_removes_listener_and_leaves_job_running():
    async def check():
        key='b'*32
        async def connected():return False
        request=SimpleNamespace(state=SimpleNamespace(owner='owner'),is_disconnected=connected)
        main.JOBS[key]={'_owner':'owner','status':'running','stage':'running'}
        stream=main.stream_job(key,request)
        await anext(stream)
        assert key in main.LISTENERS
        await stream.aclose()
        assert not main.LISTENERS
        assert main.JOBS[key]['status']=='running'
    asyncio.run(check())


def test_actual_asgi_middleware_sends_partial_before_completion():
    async def check():
        key='c'*32;sid='d'*32
        main.JOBS[key]={'id':key,'_owner':sid,'status':'running','stage':'waiting_worker'}
        cookie='mq_session='+sid+'.'+main.sign(sid)
        scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},
            'http_version':'1.1','method':'GET','scheme':'http','root_path':'',
            'path':'/api/jobs/'+key+'/events','raw_path':('/api/jobs/'+key+'/events').encode(),
            'query_string':b'','headers':[(b'cookie',cookie.encode())],
            'server':('testserver',80),'client':('testclient',1234)}
        requested=False;finished=asyncio.Event();observed=[]
        async def receive():
            nonlocal requested
            if not requested:
                requested=True
                return {'type':'http.request','body':b'','more_body':False}
            await finished.wait()
            return {'type':'http.disconnect'}
        async def send(message):
            body=message.get('body',b'')
            if not body.startswith(b'data: '):return
            payload=json.loads(body.removeprefix(b'data: '))
            observed.append(payload['stage'])
            if payload['stage']=='waiting_worker':
                main.update_job(key,stage='waiting_cpu',partial={'engines':{'quantum':{}},'complete':False})
            elif payload['stage']=='waiting_cpu':
                assert main.JOBS[key]['status']=='running'
                assert set(payload['partial']['engines'])=={'quantum'}
                main.update_job(key,status='completed',stage='completed',result={'complete':True})
        await asyncio.wait_for(main.app(scope,receive,send),2)
        assert observed==['waiting_worker','waiting_cpu','completed']
        assert not main.LISTENERS
    asyncio.run(check())


def test_stream_endpoint_requires_same_owner_and_sends_terminal_snapshot(monkeypatch):
    monkeypatch.setenv('MODAL_TOKEN_ID','test');monkeypatch.setenv('MODAL_TOKEN_SECRET','test')
    monkeypatch.setattr(main.threading,'Thread',Mock())
    with TestClient(main.app) as a,TestClient(main.app) as b:
        a.get('/api/info');b.get('/api/info')
        r=a.post('/api/jobs',headers={'X-Metaquant-Request':'1'},json={
            'request_id':str(uuid4()),'spec':{'kind':'channels','size':4,'seed':3}})
        key=r.json()['id']
        assert b.get('/api/jobs/'+key+'/events').status_code==404
        main.update_job(key,status='cancelled',stage='cancelled')
        response=a.get('/api/jobs/'+key+'/events')
        assert response.status_code==200
        assert response.headers['content-type'].startswith('text/event-stream')
        assert response.headers['x-accel-buffering']=='no'
        assert response.headers['cache-control']=='no-store'
        data=json.loads(response.text.removeprefix('data: '))
        assert data['status']=='cancelled' and '_owner' not in data
        assert not main.LISTENERS


def test_phase_history_keeps_steps_that_finish_between_polls(monkeypatch):
    ticks=iter([10,10,12,14,15])
    monkeypatch.setattr('optimizer.wire.time.perf_counter',lambda:next(ticks))
    packets=[];p=ProgressPublisher(packets.append)
    p.stage('initializing');p.stage('warming_up');p.stage('running')
    p.partial({'engines':{'quantum':{}},'complete':False})
    last=packets[-1]
    assert last['sequence']==4 and last['stage']=='waiting_cpu'
    assert [(x['stage'],x['worker_ms']) for x in last['worker_timeline']]==[
        ('initializing',0),('warming_up',2000),('running',4000),('waiting_cpu',5000)]
    assert packets[0]['worker_timeline']==[{'stage':'initializing','worker_ms':0}]


def test_cpu_process_can_run_without_loading_gpu_libraries():
    code='''
import sys,threading
from optimizer.live import initialize,bundle
from optimizer.problems import Spec,build_problem
initialize('traditional',threading.Event())
for kind,variant in [('channels','standard'),('custom','dense')]:
    p=build_problem(Spec(kind=kind,variant=variant,size=12,seed=3))
    r=bundle('traditional',p,10)
    assert len(r['trials'])==3 and r['trial_workers']==3
    assert r['energy']==p.energy(r['bits'])
assert 'torch' not in sys.modules and 'simulated_bifurcation' not in sys.modules
'''
    subprocess.run([sys.executable,'-c',code],check=True,timeout=20,
        cwd=Path(__file__).resolve().parents[1],env=os.environ.copy(),capture_output=True,text=True)
