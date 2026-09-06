import torch
import pytest
from cftn_v3.local_specialist import LocalMathTower,MathTokenizer,save_specialist,install_specialist
from cftn_v3.model import CFTN,ByteTokenizer
from cftn_v3.config import Config,TOWERS
from cftn_v3.contracts import Call,ExecutionPlan
from cftn_v3.artifact import save_bundle,load_bundle

SPEC={'hidden_size':32,'layers':2,'attention_heads':4,'feed_forward_size':64,
      'dropout':0.,'max_sequence_length':128,'receiver_layers':[0,1],'answer_min':-512,'answer_max':512}


def test_cached_decode_matches_full_forward():
    model=LocalMathTower(SPEC).eval();tok=MathTokenizer();prompt='{"op":"add","operands":[2,3]}'
    ids=torch.tensor([tok.prefix(prompt)]);expected=[]
    with torch.no_grad():
        for _ in range(8):
            t=int(model(ids)[:,-1].argmax(-1))
            if t==2:break
            expected.append(t);ids=torch.cat((ids,torch.tensor([[t]])),1)
    assert model.generate(prompt,8)[0]==tok.decode(expected)


def test_assembly_is_self_contained_and_preserves_native_weights(tmp_path):
    specialist=LocalMathTower(SPEC)
    p=tmp_path/'math.specialist';save_specialist(p,specialist,{'accepted':False})
    model=CFTN(Config.tiny(),ByteTokenizer())
    with pytest.raises(ValueError):install_specialist(model,p)
    install_specialist(model,p,allow_experimental=True)
    assert 'math' not in model.config.active
    path=tmp_path/'unified.cftn';save_bundle(path,model)
    restored,_,_=load_bundle(path)
    for name,t in specialist.state_dict().items():assert torch.equal(t,restored.towers['math'].state_dict()[name])
    assert restored.tokenizer_for('math').vocab_size==260


def test_unselected_towers_do_not_execute():
    model=CFTN(Config.tiny(),ByteTokenizer()).eval()
    calls={t:0 for t in TOWERS};hooks=[]
    for name,tower in model.towers.items():
        hooks.append(tower.embedding.register_forward_hook(lambda m,a,o,n=name:calls.__setitem__(n,calls[n]+1)))
    plan=ExecutionPlan([Call('code',0,'Write solve(x).'),Call('formal_logic',0,'P implies Q.')])
    model.generate('Use code and logic.',plan=plan,max_tokens=2)
    assert calls['code']==calls['formal_logic']==1
    assert all(v==0 for k,v in calls.items() if k not in ('code','formal_logic'))
    assert {c['tower'] for c in model.last_execution_trace}=={'code','formal_logic'}
    calls.update({t:0 for t in TOWERS})
    model.generate('Language only.',max_tokens=2)
    assert not any(calls.values())
    for h in hooks:h.remove()


def test_inactive_tower_is_never_selected():
    model=CFTN(Config.tiny(),ByteTokenizer()).eval();model.config.active=('math',)
    with torch.no_grad():
        model.dispatcher.wake_gates.weight.zero_();model.dispatcher.wake_gates.bias.fill_(20)
        model.dispatcher.dependency_head.weight.zero_();model.dispatcher.dependency_head.bias.fill_(-20)
    assert [c.tower for c in model.route('test').calls]==['math']
