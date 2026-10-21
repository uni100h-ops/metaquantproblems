"""Use Modal's actual serializer with no Torch/GPU runtime installed."""
import sys
import types
from unittest.mock import Mock
import pytest
from modal._serialization import serialize,deserialize
from modal.exception import DeserializationError
from optimizer.wire import plain_data,ProgressPublisher


def test_worker_library_str_subclass_does_not_require_library_on_render(monkeypatch):
    # Same relevant contract as torch.torch_version.TorchVersion: importable str subclass.
    module=types.ModuleType('gpu_worker_only_library')
    Version=type('Version',(str,),{'__module__':module.__name__})
    module.Version=Version
    monkeypatch.setitem(sys.modules,module.__name__,module)
    payload={'hardware':{'torch_version':Version('2.6.0+cu124')},'engines':{'quantum':{'bits':[0,1]}}}
    raw=serialize(payload)
    clean=serialize(plain_data(payload))
    writes=[];publisher=ProgressPublisher(lambda x:writes.append(serialize(x)))
    publisher.partial(payload)
    final=serialize(publisher.complete(payload))
    monkeypatch.delitem(sys.modules,module.__name__)
    with pytest.raises(DeserializationError):deserialize(raw,None)
    assert deserialize(clean,None)==deserialize(final,None)
    for packet in writes:
        decoded=deserialize(packet,None)
        assert type(decoded['partial']['hardware']['torch_version']) is str
        assert decoded['partial']['engines']['quantum']['bits']==[0,1]


@pytest.mark.parametrize('first',['traditional','quantum'])
def test_first_result_is_published_before_second_and_retained_across_stages(first):
    second='quantum' if first=='traditional' else 'traditional'
    packets=[];publisher=ProgressPublisher(packets.append)
    publisher.stage('running')
    partial={'engines':{first:{'energy':1}},'complete':False}
    publisher.partial(partial)
    first_packet=packets[-1]
    assert set(first_packet['partial']['engines'])=={first}
    publisher.stage('verifying')
    assert packets[-1]['partial']==partial
    final={'engines':{first:{'energy':1},second:{'energy':2}},'complete':True}
    assert publisher.complete(final)==final
    assert packets[-1]['stage']=='completed'
    assert set(first_packet['partial']['engines'])=={first}


def test_nonfinite_or_nondata_payload_is_rejected():
    with pytest.raises(ValueError):plain_data({'energy':float('nan')})
    with pytest.raises(TypeError):plain_data({'data':object()})


def test_reused_worker_initializes_once_and_emits_each_engine(monkeypatch):
    import modal_app
    import optimizer.live
    fake_engines=types.ModuleType('optimizer.engines')
    fake_engines.verify_gpu=Mock()
    def compare(problem,seed,progress,partial):
        partial({'engines':{'quantum':{'energy':1}},'complete':False})
        progress('verifying')
        return {'engines':{'quantum':{'energy':1},'traditional':{'energy':2}},'complete':True}
    fake_engines.compare=compare
    monkeypatch.setitem(sys.modules,'optimizer.engines',fake_engines)
    prepare=Mock();monkeypatch.setattr(optimizer.live,'prepare',prepare)
    updates=Mock();monkeypatch.setattr(modal_app,'updates',updates)
    monkeypatch.setattr(modal_app,'_worker_ready',False)
    spec={'kind':'channels','size':4,'seed':3}
    for job_id in ('a'*32,'b'*32):
        result=modal_app.compare_job.local(job_id,spec,100)
        assert result['complete'] is True and result['provider']=='Modal'
    fake_engines.verify_gpu.assert_called_once();prepare.assert_called_once()
    first=[call.args[1] for call in updates.put.call_args_list if call.args[0]=='a'*32]
    second=[call.args[1] for call in updates.put.call_args_list if call.args[0]=='b'*32]
    assert first[0]['stage']=='initializing'
    assert second[0]['stage']=='preparing_problem'
    assert 'warming_up' not in [p['stage'] for p in second]
    for packets in (first,second):
        verification=next(p for p in packets if p['stage']=='verifying')
        assert set(verification['partial']['engines'])=={'quantum'}
