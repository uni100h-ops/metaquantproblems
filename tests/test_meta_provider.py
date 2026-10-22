"""Provider failures remain actionable and never expose a key or prompt."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import httpx
import pytest
from server import main, meta_ai


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv('META_API_KEY','fw_private_test_key')
    monkeypatch.setenv('META_MODEL','accounts/fireworks/models/muse-spark-open')
    monkeypatch.setenv('META_API_BASE_URL','https://api.fireworks.ai/inference/v1')
    client=Mock()
    client.__enter__=Mock(return_value=client)
    client.__exit__=Mock(return_value=False)
    monkeypatch.setattr(meta_ai.httpx,'Client',lambda **kwargs:client)
    return client


@pytest.mark.parametrize('status,code',[
    (400,'meta_request_rejected'),(401,'meta_auth'),(402,'meta_quota'),
    (403,'meta_access'),(404,'meta_model_unavailable'),(412,'meta_account'),
    (429,'meta_rate_limit'),(500,'meta_provider_unavailable'),(504,'meta_timeout')])
def test_provider_failure_category_and_no_sensitive_logging(provider,caplog,status,code):
    provider.post.return_value=httpx.Response(status,json={'error':{'message':'fw_private_test_key private prompt'}})
    with pytest.raises(meta_ai.AIError) as failure:
        meta_ai.chat('system','private prompt')
    assert failure.value.detail()=={'code':code,'provider_status':status}
    assert f'provider_http={status}' in caplog.text
    assert 'fw_private_test_key' not in caplog.text
    assert 'private prompt' not in caplog.text
    assert provider.post.call_count==1


def test_success_uses_configured_model_and_returns_only_answer(provider):
    provider.post.return_value=httpx.Response(200,json={'choices':[
        {'message':{'content':'Measured result explanation.','reasoning_content':'private reasoning'}}]})
    text,meta=meta_ai.chat('system','user')
    assert text=='Measured result explanation.' and meta['api_called'] is True
    assert meta['model']=='accounts/fireworks/models/muse-spark-open'
    assert provider.post.call_args.kwargs['json']['model']==meta['model']
    assert provider.post.call_args.kwargs['json']['reasoning_effort']=='low'
    assert provider.post.call_args.kwargs['json']['max_tokens']==4096


def test_truncated_reasoning_is_not_presented_as_an_explanation(provider):
    provider.post.return_value=httpx.Response(200,json={'choices':[{
        'finish_reason':'length','message':{'content':'','reasoning_content':'internal analysis'}}]})
    with pytest.raises(meta_ai.AIError,match='^meta_output_limit$'):meta_ai.chat('s','u')


def test_explicit_llama_configuration_keeps_compatible_parameters(provider,monkeypatch):
    monkeypatch.setenv('META_MODEL','accounts/fireworks/models/llama-v3p3-70b-instruct')
    provider.post.return_value=httpx.Response(200,json={'choices':[{'message':{'content':'OK'}}]})
    meta_ai.chat('s','u')
    assert 'reasoning_effort' not in provider.post.call_args.kwargs['json']
    assert provider.post.call_args.kwargs['json']['max_tokens']==2000


def test_endpoint_paste_and_whitespace_are_normalized(provider,monkeypatch):
    monkeypatch.setenv('META_API_BASE_URL',' https://api.fireworks.ai/inference/v1/chat/completions/ ')
    monkeypatch.setenv('META_MODEL',' accounts/fireworks/models/muse-spark-open ')
    provider.post.return_value=httpx.Response(200,json={'choices':[{'message':{'content':'OK'}}]})
    meta_ai.chat('s','u')
    assert provider.post.call_args.args[0]=='https://api.fireworks.ai/inference/v1/chat/completions'
    assert meta_ai.configured()


def test_network_timeout_distinguished(provider):
    provider.post.side_effect=httpx.ReadTimeout('sensitive request')
    with pytest.raises(meta_ai.AIError,match='^meta_timeout$'):meta_ai.chat('s','u')


def test_empty_answer_never_passes_as_success(provider):
    provider.post.return_value=httpx.Response(200,json={'choices':[{'message':{'content':None}}]})
    with pytest.raises(meta_ai.AIError,match='^meta_format$'):meta_ai.chat('s','u')


def test_explain_keeps_result_and_reports_provider_status_then_caches_success(monkeypatch):
    main.JOBS.clear();main.RECENT.clear()
    key='e'*32
    result={'complete':True,'engines':{'quantum':{'energy':7}}}
    main.JOBS[key]={'_owner':'test-owner','status':'completed','result':result,'_language':'en'}
    req=SimpleNamespace(state=SimpleNamespace(owner='test-owner'))
    fake=Mock(side_effect=[meta_ai.AIError('meta_model_unavailable',404),{'text':'Done','meta':{'api_called':True}}])
    monkeypatch.setattr(meta_ai,'explain',fake)
    try:
        with pytest.raises(main.HTTPException) as failure:main.explain(key,req)
        assert failure.value.detail=={'code':'meta_model_unavailable','provider_status':404}
        assert main.JOBS[key]['result'] is result
        assert not main.PLANNING.locked()
        assert main.explain(key,req)['text']=='Done'
        assert main.explain(key,req)['text']=='Done'
        assert fake.call_count==2
    finally:
        main.JOBS.clear();main.RECENT.clear()
