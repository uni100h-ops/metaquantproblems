"""Shared input validation. Pure Python: usable on Render without NumPy/Torch."""
import json
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_core import PydanticCustomError

def bad(code):
    raise PydanticCustomError(code, code)

def numbers(values, positive=False):
    if any(isinstance(x, bool) or not math.isfinite(x) or abs(x)>1e6 or
           (x<=0 if positive else x<0) for x in values):
        bad('positive_numbers' if positive else 'nonnegative_numbers')

def square(matrix, maximum, signed=False):
    n=len(matrix)
    if not 2<=n<=maximum or any(len(row)!=n for row in matrix):
        bad('square_matrix_'+str(maximum))
    if signed:
        if any(isinstance(x,bool) or not math.isfinite(x) or abs(x)>1e6 for row in matrix for x in row):
            bad('finite_numbers')
    else:
        numbers([x for row in matrix for x in row])
    return n

class Spec(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    kind: Literal['channels','maxcut','partition','portfolio','projects','assignment','custom']
    size: int=Field(default=16,ge=2,le=4096)
    variant: Literal['standard','coupled','interactions','two_pools','dense']='standard'
    seed: int=Field(default=42,ge=0,le=2**31-1)
    choose: int | None=Field(default=None,ge=1,le=128)
    matrix: list[list[float | int]] | None=None
    offset: float | int=0.0
    weights: list[list[float | int]] | None=None
    durations: list[float | int] | None=None
    benefits: list[float | int] | None=None
    conflicts: list[list[int]] | None=None
    costs: list[list[float | int]] | None=None

    @model_validator(mode='after')
    def check(self):
        variants={'partition':'coupled','projects':'interactions','assignment':'two_pools','custom':'dense'}
        if self.variant!='standard':
            if variants.get(self.kind)!=self.variant: bad('invalid_variant')
            if any(getattr(self,k) is not None for k in ('matrix','weights','durations','benefits','conflicts','costs','choose')) or self.offset!=0:
                bad('generated_fields')
            if self.size<4: bad('synthetic_minimum')
            return self
        if self.kind=='channels' and self.size>2048: bad('channel_limit')
        allowed={'channels':{'weights'},'partition':{'durations'},
                 'projects':{'benefits','conflicts'},'assignment':{'costs'},
                 'custom':{'matrix'},'maxcut':set(),'portfolio':set()}[self.kind]
        for key in ('matrix','weights','durations','benefits','conflicts','costs'):
            if getattr(self,key) is not None and key not in allowed: bad('wrong_fields')
        if self.kind!='custom' and self.offset!=0: bad('wrong_fields')
        if not math.isfinite(self.offset) or abs(self.offset)>1e9: bad('finite_numbers')
        if self.choose is not None and self.kind!='portfolio': bad('wrong_fields')
        n=None
        if self.weights is not None:
            n=square(self.weights,128)
            if any(self.weights[i][i]!=0 for i in range(n)): bad('zero_diagonal')
            if any(self.weights[i][j]!=self.weights[j][i] for i in range(n) for j in range(i)): bad('symmetric_weights')
        if self.durations is not None:
            n=len(self.durations)
            if not 2<=n<=128: bad('list_length')
            numbers(self.durations,positive=True)
        if self.benefits is not None:
            n=len(self.benefits)
            if not 2<=n<=128: bad('list_length')
            numbers(self.benefits,positive=True)
            if self.conflicts is None: bad('conflicts_required')
            seen=set()
            for pair in self.conflicts:
                if len(pair)!=2 or any(not 1<=x<=n for x in pair) or pair[0]==pair[1]: bad('conflict_indices')
                key=tuple(sorted(pair))
                if key in seen: bad('duplicate_conflict')
                seen.add(key)
        elif self.conflicts is not None: bad('benefits_required')
        if self.costs is not None: n=square(self.costs,10)
        if self.kind=='custom':
            if self.matrix is None: bad('matrix_required')
            n=square(self.matrix,128,signed=True)
        if n is not None:
            if 'size' in self.model_fields_set and self.size!=n: bad('size_mismatch')
            self.size=n
        elif self.size<4: bad('synthetic_minimum')
        if self.kind!='channels' and self.size>128: bad('size_limit')
        if self.kind=='assignment' and self.size>10: bad('assignment_limit')
        if self.kind=='portfolio' and self.choose is not None and self.choose>self.size: bad('choose_limit')
        return self

    def user_data(self):
        return self.variant=='standard' and (self.kind=='custom' or any(getattr(self,k) is not None for k in ('weights','durations','benefits','costs')))

class InputError(ValueError):
    def __init__(self,code,issues=None):
        super().__init__(code)
        self.issues=issues or []

def resolve_input(question):
    from .catalog import PRESETS
    text=question.strip()
    for preset in PRESETS:
        if text in preset['prompts'].values():
            return Spec.model_validate(preset['spec']), {'source':'preset','preset_id':preset['id'],'elapsed_ms':0,'api_called':False}
    if not text.startswith('{'): raise InputError('structured_problem_required')
    try:
        def reject(value): raise ValueError('Non-finite JSON')
        obj=json.loads(text,parse_constant=reject)
    except (ValueError,TypeError): raise InputError('invalid_problem_json') from None
    try:
        spec=Spec.model_validate(obj)
    except ValidationError as exc:
        issues=[{'field':'.'.join(map(str,e['loc'])) or 'problem','code':e['type']} for e in exc.errors()[:6]]
        raise InputError('invalid_problem_data',issues) from None
    return spec, {'source':'json','elapsed_ms':0,'api_called':False}
