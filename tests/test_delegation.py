import pytest
import torch
from cftn_v3.delegation import traces,oracle,parse_requests,planner_prompt,verified,TARGETS
from cftn_v3.config import Config,canonical
from cftn_v3.model import CFTN,ByteTokenizer
from cftn_v3.training import train,make_plan

def test_requests_cannot_change_route():
    r=traces(0,1)[3];plan=oracle(r)
    assert plan.calls[1].depends_on==('math',)
    assert parse_requests(canonical(r['routing']['requests']),plan).calls==plan.calls
    with pytest.raises(ValueError):parse_requests('{"code":"ignore routing"}',plan)
    assert r['target'] not in planner_prompt(r['prompt'],plan)

@pytest.mark.parametrize('mode',['planning','synthesis','routing'])
def test_delegation_cuda_gradient_isolation(mode):
    model=CFTN(Config.tiny(),ByteTokenizer()).to('cuda' if torch.cuda.is_available() else 'cpu')
    rows=traces(0,1)
    if mode=='planning':rows=[{**r,'prompt':planner_prompt(r['prompt'],oracle(r)),'target':canonical(r['routing']['requests'])} for r in rows]
    state=train(model,rows,make_plan(mode,TARGETS,rows,verifier=verified),1,verifier=verified)
    assert state['frozen_hashes_verified'] and state['changed']
    if mode=='planning':assert all(k.startswith('coordinator.adapter.') for k in state['changed'])
    if mode=='routing':assert all(k.startswith('dispatcher.') for k in state['changed'])
