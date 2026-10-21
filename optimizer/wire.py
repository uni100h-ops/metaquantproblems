"""Plain-data boundary: Render does not install the GPU worker's libraries."""
import json


def plain_data(value):
    # JSON round-trip strips str/float subclasses (notably torch.TorchVersion).
    # Modal otherwise pickles their classes, requiring torch/numpy on Render.
    return json.loads(json.dumps(value,allow_nan=False))


class ProgressPublisher:
    """Keep the latest completed engines when subsequent stages are published."""
    def __init__(self,write):
        self.write=write
        self.packet={}

    def stage(self,name):
        self.packet['stage']=name
        self.write(plain_data(self.packet))

    def partial(self,result):
        self.packet['partial']=plain_data(result)
        self.stage('running')

    def complete(self,result):
        result=plain_data(result)
        self.packet['partial']=result
        self.stage('completed')
        return result
