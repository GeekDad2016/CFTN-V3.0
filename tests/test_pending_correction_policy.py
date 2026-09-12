import json
import pytest
import torch
from cftn_v3.pending_correction_policy import apply

@pytest.mark.parametrize('selected,enabled',[('targeted',1),('baseline',0),('start',0)])
def test_post_trial_handoff_preserves_weights_and_applies_conditional_policy(tmp_path,selected,enabled):
    root=tmp_path;out=root/'math';out.mkdir();exp=root/'generated_correction_probe';exp.mkdir()
    (exp/'decision.json').write_text(json.dumps({'selected':selected}))
    (root/'pending_correction_policy.json').write_text(json.dumps({'sigreg_coefficient':.0003,'validation_loss_threshold':.003}))
    payload={'weights':{'x':torch.tensor([2.])},'optimizer':{'state':{}},'metadata':{'round':12,'cursor':1,'policy':{},'controller':{}}}
    torch.save(payload,out/'current.specialist')
    cfg={'queue':[{'tower':'math','output':str(out),'policy':{}},{'tower':'string','policy':{'lr':.001}}]}
    config=root/'config.json';config.write_text(json.dumps(cfg))
    result=apply(config,cfg,root)
    saved=torch.load(out/'current.specialist',weights_only=True)
    assert saved['metadata']['policy']['generated_correction']==enabled
    assert saved['metadata']['policy']['sigreg_coefficient']==.0003
    assert not saved['metadata']['controller']['validation_enabled']
    assert saved['metadata']['cursor']==1
    assert torch.equal(saved['weights']['x'],payload['weights']['x'])
    assert result['queue'][1]==cfg['queue'][1]
    assert (out/'before_stage_corrections_round_12.specialist').exists()
    assert not (root/'pending_correction_policy.json').exists()
