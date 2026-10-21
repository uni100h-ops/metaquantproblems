import json
from uuid import uuid4
from unittest.mock import MagicMock
import numpy as np
import pytest
from fastapi.testclient import TestClient
from optimizer.catalog import PRESETS
from optimizer.specs import Spec,resolve_input,InputError
from optimizer.problems import build_problem
from optimizer.dense_engines import classical_dense
from server import main,meta_ai

HEAD={'X-Metaquant-Request':'1'}
@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setenv('COOKIE_SECURE','0')
    monkeypatch.delenv('MODAL_TOKEN_ID',raising=False)
    monkeypatch.delenv('MODAL_TOKEN_SECRET',raising=False)
    monkeypatch.delenv('META_API_KEY',raising=False)
    main.JOBS.clear();main.RECENT.clear();main.ACTIVE=None

@pytest.mark.parametrize('preset',PRESETS,ids=lambda p:p['id'])
def test_catalog(preset):
    s=Spec.model_validate(preset['spec'])
    assert s.size in (512,1024,2048,3072,4096)
    assert s.seed>=20261026
    assert set(preset['prompts'])=={'en','es','fr','de'}

@pytest.mark.parametrize('kind,variant',[('partition','coupled'),('projects','interactions'),('assignment','two_pools'),('custom','dense')])
def test_native_cpu_energy(kind,variant):
    p=build_problem(Spec(kind=kind,variant=variant,size=12,seed=71))
    row=classical_dense(p,100)
    x=np.asarray(row['bits'])
    assert row['energy']==pytest.approx(float(x@p.q@x+p.offset))
    assert row['independent_cpu_energy']==pytest.approx(row['energy'])
    assert set(row['bits'])<={0,1}

def test_real_data_and_invalid_data():
    s,_=resolve_input('{"kind":"custom","matrix":[[-1,2],[2,-3]]}')
    assert s.size==2 and s.user_data()
    p=build_problem(s)
    assert p.energy([1,1])==pytest.approx(0)
    for text in ['hello','{"kind":"custom","matrix":[[NaN,0],[0,1]]}','{"kind":"channels","weights":[[0,2],[3,0]]}']:
        with pytest.raises(InputError):resolve_input(text)

def test_http_without_keys():
    with TestClient(main.app) as c:
        assert c.get('/').status_code==200
        r=c.get('/api/info');assert not r.json()['gpu_configured'] and not r.json()['meta_configured']
        assert 'Content-Security-Policy' in r.headers
        assert c.post('/api/validate',json={'question':'{}'}).status_code==403
        assert c.post('/api/validate',headers=HEAD,json={'question':'{"kind":"partition","durations":[2,3]}'}).status_code==200
        assert c.post('/api/jobs',headers=HEAD,json={'request_id':str(uuid4()),'spec':PRESETS[0]['spec']}).status_code==503

def test_job_ownership_idempotency(monkeypatch):
    monkeypatch.setenv('MODAL_TOKEN_ID','test');monkeypatch.setenv('MODAL_TOKEN_SECRET','test')
    monkeypatch.setattr(main.threading,'Thread',MagicMock())
    with TestClient(main.app) as a,TestClient(main.app) as b:
        a.get('/api/info');b.get('/api/info')
        body={'request_id':str(uuid4()),'spec':PRESETS[0]['spec']}
        r=a.post('/api/jobs',headers=HEAD,json=body);assert r.status_code==200
        key=r.json()['id'];assert '_owner' not in r.json()
        assert a.post('/api/jobs',headers=HEAD,json=body).status_code==200
        assert b.get('/api/jobs/'+key).status_code==404
        assert b.post('/api/jobs/'+key+'/cancel',headers=HEAD,json={}).status_code==404
        assert a.post('/api/jobs/'+key+'/cancel',headers=HEAD,json={}).json()['stage']=='cancelling'

def test_modal_timeout_polls_instead_of_failing(monkeypatch):
    import modal
    key=uuid4().hex;main.ACTIVE=key
    main.JOBS[key]={'spec':PRESETS[0]['spec'],'status':'running','_cancel':False}
    call=MagicMock();call.get.side_effect=[modal.exception.TimeoutError(),{'ok':True}]
    fn=MagicMock();fn.spawn.return_value=call
    updates=MagicMock();updates.get.return_value={'stage':'running','partial':{'complete':False}}
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**kw:fn)
    monkeypatch.setattr(modal.Dict,'from_name',lambda *a,**kw:updates)
    monkeypatch.setattr(main,'valid_result',lambda x:x=={'ok':True})
    monkeypatch.setattr(main.time,'sleep',lambda _:None)
    main.run_job(key)
    assert main.JOBS[key]['status']=='completed'
    assert main.JOBS[key]['partial']=={'complete':False}
    assert main.ACTIVE is None
    call.cancel.assert_not_called()

def test_model_contract(monkeypatch):
    monkeypatch.setattr(meta_ai,'chat',lambda *a:('{"status":"ready","message":"OK","spec":{"kind":"partition","durations":[2,3]}}',{'api_called':True}))
    assert meta_ai.formulate('balance','es')['spec']['size']==2
    monkeypatch.setattr(meta_ai,'chat',lambda *a:('{"status":"ready","message":"OK","spec":{"kind":"maxcut"}}',{}))
    with pytest.raises(meta_ai.AIError):meta_ai.formulate('bad','es')

def test_reject_unverified_gpu():
    assert not main.valid_result({'complete':True})
