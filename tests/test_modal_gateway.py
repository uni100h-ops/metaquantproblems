"""Regression checks with a cold Modal worker; no cloud calls or compute charges."""
import logging
from uuid import uuid4
from unittest.mock import Mock

import modal
import pytest
from server import main


def verified_result():
    evidence={"cpu_fallback":False,"matrix_device":"cuda:0",
              "solution_device":"cuda:0","objective_device":"cuda:0"}
    return {"schema_version":"metaquant-forge-v1","complete":True,
            "hardware":{"sb_gpu_vendor":"NVIDIA CUDA","gpu_execution_verified":True,
                        "cpu_fallback":False,"cuda_version":"test-fixture"},
            "engines":{"traditional":{"trials":[{}, {}, {}]},
                       "quantum":{**evidence,"trials":[evidence.copy() for _ in range(3)]}}}


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.delenv('MODAL_ENVIRONMENT',raising=False)
    main.JOBS.clear();main.ACTIVE=None
    key=uuid4().hex
    main.JOBS[key]={'spec':{'seed':12},'status':'running','_cancel':False}
    main.ACTIVE=key
    call=Mock();call.get.side_effect=[modal.exception.TimeoutError(),verified_result()]
    fn=Mock();fn.spawn.return_value=call
    progress=Mock();progress.get.return_value=None
    function_lookup=Mock(return_value=fn)
    dict_lookup=Mock(return_value=progress)
    monkeypatch.setattr(modal.Function,'from_name',function_lookup)
    monkeypatch.setattr(modal.Dict,'from_name',dict_lookup)
    monkeypatch.setattr(main.time,'sleep',lambda _:None)
    yield key,fn,call,progress,function_lookup,dict_lookup
    main.JOBS.clear();main.ACTIVE=None


def test_first_run_survives_missing_progress_dict_during_cold_start(worker):
    key,fn,call,progress,_,dict_lookup=worker
    state={'exists':False,'create_requested':False}
    def lookup(*args,**kwargs):
        state['create_requested']=kwargs.get('create_if_missing',False)
        return progress
    def hydrate():
        if state['create_requested']:state['exists']=True
        if not state['exists']:raise modal.exception.NotFoundError('Progress Dict does not exist yet')
    def get(*args):
        hydrate()
        return {'stage':'warming_up'}
    dict_lookup.side_effect=lookup
    progress.hydrate.side_effect=hydrate
    progress.get.side_effect=get
    main.run_job(key)
    assert main.JOBS[key]['status']=='completed'
    assert main.JOBS[key]['result']==verified_result()
    assert main.ACTIVE is None
    call.cancel.assert_not_called()


def test_progress_read_failure_does_not_cancel_valid_computation(worker):
    key,_,call,progress,_,_=worker
    progress.get.side_effect=modal.exception.NotFoundError('Progress temporarily unavailable')
    main.run_job(key)
    assert main.JOBS[key]['status']=='completed'
    call.cancel.assert_not_called()


@pytest.mark.parametrize('environment',[None,'main','preview'])
def test_both_resources_use_same_environment(worker,monkeypatch,environment):
    key,_,_,_,function_lookup,dict_lookup=worker
    if environment is not None:monkeypatch.setenv('MODAL_ENVIRONMENT',environment)
    main.run_job(key)
    assert function_lookup.call_args.kwargs['environment_name']==(environment or 'main')
    assert dict_lookup.call_args.kwargs['environment_name']==(environment or 'main')


def test_missing_app_is_identified_before_spawning(worker,monkeypatch,caplog):
    key,fn,call,_,_,_=worker
    secret='TEST-CREDENTIAL-NOT-A-REAL-TOKEN'
    monkeypatch.setenv('MODAL_TOKEN_SECRET',secret)
    fn.hydrate.side_effect=modal.exception.NotFoundError('No app: '+secret)
    with caplog.at_level(logging.ERROR,logger='metaquant'):main.run_job(key)
    assert main.JOBS[key]['error']=='modal_app_not_found'
    assert 'lookup_function' in caplog.text and '[REDACTED]' in caplog.text
    assert secret not in caplog.text and secret not in str(main.snapshot(main.JOBS[key]))
    fn.spawn.assert_not_called();call.cancel.assert_not_called()
    assert main.ACTIVE is None


def test_denied_access_does_not_start_compute(worker):
    key,fn,_,progress,_,_=worker
    progress.hydrate.side_effect=modal.exception.PermissionDeniedError('Denied')
    main.run_job(key)
    assert main.JOBS[key]['error']=='modal_access_denied'
    fn.spawn.assert_not_called()
    assert main.ACTIVE is None


def test_worker_timeout_stops_polling(worker):
    key,_,call,_,_,_=worker
    call.get.side_effect=modal.exception.FunctionTimeoutError('Worker timed out')
    main.run_job(key)
    assert main.JOBS[key]['error']=='job_timeout'
    assert call.get.call_count==1
    assert main.ACTIVE is None


def test_unverified_gpu_result_still_rejected(worker):
    key,_,call,_,_,_=worker
    call.get.side_effect=[{'complete':True}]
    main.run_job(key)
    assert main.JOBS[key]['status']=='failed'
    call.cancel.assert_called_once()


def test_cancelled_job_stops_worker(worker):
    key,_,call,_,_,_=worker
    main.JOBS[key]['_cancel']=True
    main.run_job(key)
    assert main.JOBS[key]['status']=='cancelled'
    call.get.assert_not_called()
    call.cancel.assert_called_once_with(terminate_containers=True)


def test_completed_engine_preserved_even_when_other_engine_fails(worker):
    key,_,call,progress,_,_=worker
    partial={'engines':{'quantum':{'energy':12}},'complete':False}
    progress.get.return_value={'stage':'running','partial':partial}
    call.get.side_effect=RuntimeError('CPU failed')
    main.run_job(key)
    assert main.JOBS[key]['status']=='failed'
    assert main.JOBS[key]['partial']==partial


def test_outdated_worker_serialization_failure_has_specific_error(worker):
    key,_,call,_,_,_=worker
    call.get.side_effect=modal.exception.DeserializationError('Missing worker library')
    main.run_job(key)
    assert main.JOBS[key]['error']=='result_transfer_failed'
