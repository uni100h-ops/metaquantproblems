"""Plain-data boundary: Render does not install the GPU worker's libraries."""
import json
import time


def plain_data(value):
    # JSON round-trip strips str/float subclasses (notably torch.TorchVersion).
    # Modal otherwise pickles their classes, requiring torch/numpy on Render.
    return json.loads(json.dumps(value,allow_nan=False))


class ProgressPublisher:
    """Keep the latest completed engines when subsequent stages are published."""
    def __init__(self,write):
        self.write=write
        self.packet={}
        self.started=time.perf_counter()
        self.timeline=[]
        self.sequence=0

    def stage(self,name):
        elapsed=1000*(time.perf_counter()-self.started)
        if self.packet.get('stage')!=name:
            self.timeline.append({'stage':name,'worker_ms':round(elapsed,2)})
        self.packet['stage']=name
        self.sequence+=1
        self.packet.update(sequence=self.sequence,worker_timeline=list(self.timeline),
                           worker_elapsed_ms=round(elapsed,2))
        self.write(plain_data(self.packet))

    def partial(self,result):
        self.packet['partial']=plain_data(result)
        engines=result.get('engines',{})
        self.stage('waiting_cpu' if set(engines)=={'quantum'} else
                   'waiting_gpu' if set(engines)=={'traditional'} else 'verifying')

    def complete(self,result):
        result=plain_data(result)
        self.packet['partial']=result
        self.stage('completed')
        return result
