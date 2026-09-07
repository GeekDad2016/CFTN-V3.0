import json
from fractions import Fraction
from pathlib import Path
import pytest
import torch
from cftn_v3.full_curriculum_data import make,extensions
from cftn_v3.math_procedures import solve,score
from cftn_v3.full_curriculum_training import failed_criteria,training_rows,batches,panel
from cftn_v3.local_specialist import LocalMathTower,save_specialist,load_specialist,install_specialist
from cftn_v3.model import CFTN,ByteTokenizer
from cftn_v3.config import Config
from cftn_v3.artifact import save_bundle,load_bundle

def test_tuple_contract_rejects_comma_loss_and_duplicate_answers():
    r=make({'op':'simultaneous_solve','equations':[[1,0,2],[0,1,16]]},8,'test')
    assert r['answer']=='[2,16]'
    assert score(r['target'],r)['answer_correct']
    assert not score('<answer>216</answer>',r)['answer_correct']
    assert not score('<answer>2,16</answer><answer>216</answer>',r)['answer_correct']
    assert score('<answer>2,16</answer>',r)['answer_correct']
    assert not score(r['target'],r,False)['format_correct']

def test_extended_answers_against_independent_symbolic_math():
    sp=pytest.importorskip('sympy');x=sp.Symbol('x')
    for _,_,ir in extensions(count=12):
        answer,_=solve(ir);op=ir['op']
        if op=='quadratic_general':expected='['+','.join(str(v) for v in sorted(sp.solve(ir['a']*x*x+ir['b']*x+ir['c'],x)))+']'
        elif op=='differentiate_polynomial':expected='['+','.join(str(sp.diff(sum(c*x**i for i,c in enumerate(ir['coefficients'])),x).expand().coeff(x,i)) for i in range(len(ir['coefficients'])-1))+']'
        elif op=='definite_integral_polynomial':expected=str(sp.integrate(sum(c*x**i for i,c in enumerate(ir['coefficients'])),(x,ir['lower'],ir['upper'])))
        elif op=='matrix_det_3x3':expected=str(sp.Matrix(ir['matrix']).det())
        elif op=='eigenvalues_2x2':expected='['+','.join(str(v) for v in sorted(sp.Matrix(ir['matrix']).eigenvals()))+']'
        elif op=='polynomial_derivative_value':expected=str(sp.diff(sum(c*x**i for i,c in enumerate(ir['coefficients'])),x).subs(x,ir['point']))
        elif op=='binomial_probability':
            p=sp.Rational(*ir['probability']);n,k=ir['trials'],ir['successes'];expected=str(sp.binomial(n,k)*p**k*(1-p)**(n-k))
        elif op=='weighted_expectation':expected=str(sum(v*sp.Rational(*w) for v,w in zip(ir['values'],ir['weights'])))
        elif op=='bezout_coefficients':
            g,s,t=json.loads(answer);assert ir['value']*s+ir['modulus']*t==g==int(sp.gcd(ir['value'],ir['modulus']));continue
        elif op=='permutation_order':
            from sympy.combinatorics import Permutation
            expected=str(Permutation(ir['permutation']).order())
        elif op=='geometric_series_point':expected=str(abs(Fraction(*ir['ratio'])*Fraction(*ir['point']))<1).lower()
        elif op=='polynomial_identity':expected=str(sp.expand(sum(c*x**i for i,c in enumerate(ir['left']))-sum(c*x**i for i,c in enumerate(ir['right'])))==0).lower()
        else:raise AssertionError(op)
        assert answer==expected,(ir,answer,expected)

def test_replay_is_prior_only_and_criterion_balanced():
    active=[make({'op':'add','left':i,'right':2},2,'active') for i in range(20)]
    prior=[make({'op':'add','left':i,'right':1},0,'old_a' if i<18 else 'old_b') for i in range(20)]
    rows=training_rows(active,prior,active,['active'],['old_b'],128,1,1)
    assert sum(r['stage']==2 for r in rows)==96
    assert sum(r['stage']==0 for r in rows)==32
    assert sum(r['criterion']=='old_b' for r in rows)>=16
    assert rows==training_rows(active,prior,active,['active'],['old_b'],128,1,1)
    assert len(panel(prior,2))==4
    assert sum(len(b) for b in batches(rows))==128

def test_bad_single_criterion_blocks_gate_even_with_high_overall_score():
    report={'accuracy':.99,'criteria':{'good':{'accuracy':1.,'format_accuracy':1.,'trace_accuracy':1.},'bad':{'accuracy':.5,'format_accuracy':1.,'trace_accuracy':1.}}}
    assert failed_criteria(report)==['bad']
    report['criteria']['bad']['accuracy']=1
    assert not failed_criteria(report)

def test_resume_preserves_optimizer_rng_and_native_string_assembly(tmp_path):
    spec={'hidden_size':32,'layers':2,'attention_heads':4,'feed_forward_size':64,'dropout':0.,'max_sequence_length':128}
    tower=LocalMathTower(spec);optimizer=torch.optim.AdamW(tower.parameters(),lr=.001)
    tower(torch.tensor([[1,8,9,2]])).square().mean().backward();optimizer.step()
    p=tmp_path/'string.specialist';save_specialist(p,tower,{'tower':'string','accepted':True,'cursor':3},optimizer)
    restored,saved=load_specialist(p);assert saved['tower']=='string' and saved['metadata']['cursor']==3
    for k,v in tower.state_dict().items():assert torch.equal(v,restored.state_dict()[k])
    resumed=torch.optim.AdamW(restored.parameters());resumed.load_state_dict(saved['optimizer'])
    assert resumed.param_groups[0]['lr']==.001
    torch.set_rng_state(saved['torch_rng']);expected=torch.rand(3);torch.set_rng_state(saved['torch_rng']);assert torch.equal(expected,torch.rand(3))
    model=CFTN(Config.tiny(),ByteTokenizer());install_specialist(model,p)
    unified=tmp_path/'all.cftn';save_bundle(unified,model);loaded,_,_=load_bundle(unified)
    assert 'string' not in loaded.config.active
    assert torch.equal(loaded.towers['string'].token_embedding.weight,tower.token_embedding.weight)

@pytest.mark.parametrize('passes',[False,True])
def test_runner_recovers_or_blocks_without_skipping_stage(tmp_path,monkeypatch,passes):
    import contextlib
    from types import SimpleNamespace
    import cftn_v3.full_curriculum_training as t
    spec={'hidden_size':8,'layers':1,'attention_heads':2,'feed_forward_size':16,'dropout':0.,'max_sequence_length':128}
    model=LocalMathTower(spec);monkeypatch.setattr(model,'to',lambda *a,**k:model)
    monkeypatch.setattr(t,'load_specialist',lambda p:(model,{'tower':'math','metadata':{}}))
    monkeypatch.setattr(t.torch.cuda,'is_available',lambda:True);monkeypatch.setattr(t.torch.cuda,'get_device_name',lambda:'mock')
    monkeypatch.setattr(t.torch.cuda,'memory_allocated',lambda:0);monkeypatch.setattr(t.torch,'autocast',lambda *a,**k:contextlib.nullcontext())
    monkeypatch.setattr(t,'batch_loss',lambda m,r:m.token_embedding.weight.square().mean())
    r=make({'op':'add','left':2,'right':3},0,'addition')
    manifest={'stages':[{'name':'add','index':0,'scope':'add','remediation':'repair.jsonl'}]}
    monkeypatch.setattr(t,'verify_manifest',lambda p:manifest);monkeypatch.setattr(t,'file_hash',lambda p:'hash')
    monkeypatch.setattr(t,'read',lambda p:[r]);reports=[];saves=[]
    def evaluate(m,rows,progress=None):
        accuracy=1. if passes or not rows else 0.
        metric={'accuracy':accuracy,'format_accuracy':accuracy,'trace_accuracy':accuracy,'examples':len(rows)}
        reports.append(len(rows));return {**metric,'criteria':{'addition':metric} if rows else {},'samples':[]}
    monkeypatch.setattr(t,'evaluate',evaluate)
    monkeypatch.setattr(t,'save_specialist',lambda p,m,meta,optimizer=None:saves.append(dict(meta)))
    args=SimpleNamespace(output=str(tmp_path/'run'),data=str(tmp_path/'data'),initial_checkpoint='unused',normal_rounds=2,remediation_rounds=2,attempts=1,examples=4,lr=.001)
    t.run(args);status=json.loads((tmp_path/'run/status.json').read_text())
    assert status['state']==('complete' if passes else 'blocked')
    assert not (tmp_path/'native_training.lock').exists()
    if passes:assert saves[-1]['accepted'] and (tmp_path/'run/final_test.json').exists()
    else:
        assert status['remediation_attempt']==1
        assert len(list((tmp_path/'run').glob('*_epoch_*.json')))==4
        assert not saves[-1]['completed'] and not (tmp_path/'run/sealed_test_started.json').exists()

def test_failed_worker_never_hands_off_to_next_tower(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import cftn_v3.local_curriculum_pipeline as p
    out=tmp_path/'math';out.mkdir();(out/'status.json').write_text(json.dumps({'state':'blocked','reason':'mastery failed'}))
    cfg={'root':str(tmp_path),'queue':[{'tower':name,'output':str(tmp_path/name),'data':'unused','initial_checkpoint':'unused'} for name in ['math','string']]}
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg));calls=[]
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**k:(calls.append(a),SimpleNamespace(returncode=0))[1])
    p.run(path);assert len(calls)==1
    assert json.loads((tmp_path/'queue.json').read_text())['state']=='blocked'

def test_equivalent_arithmetic_has_one_split_group():
    from cftn_v3.expand_foundations import group_id
    assert group_id({'op':'add','left':1,'right':7})==group_id({'op':'add','operands':[7,1]})
    assert group_id({'op':'multiply','left':2,'right':9})==group_id({'op':'multiply','left':9,'right':2})
    assert group_id({'op':'subtract','left':2,'right':9})!=group_id({'op':'subtract','left':9,'right':2})
    a=make({'op':'add','left':1,'right':7},0,'addition');b=make({'op':'add','operands':[7,1]},0,'addition')
    a['split_group_id']=b['split_group_id']=group_id(a['ir'])
    assert len(panel([a,b],32))==1

def test_foundation_extensions_respect_ranges_and_answers():
    from cftn_v3.expand_foundations import candidates
    checked=set()
    for stage,criterion,ir in candidates():
        answer,_=solve(ir);op=ir['op']
        if stage==0:assert sum(ir['operands'])<=20
        if stage==1 and op=='compare_expressions':
            values=[]
            for side in ['left','right']:
                a,b=ir[side];value=a+b if ir[side+'_op']=='add' else a-b
                assert 0<=value<=20;values.append(value)
            a,b=values;assert answer==('<' if a<b else '>' if a>b else '=')
        if op=='missing_subtrahend':assert ir['left']-int(answer)==ir['result']
        if op=='missing_minuend':assert int(answer)-ir['right']==ir['result']
        if op=='compose_place_value':assert int(answer)==10*ir['tens']+ir['ones']
        if op=='compare_place_value':
            a=10*ir['tens']+ir['ones'];b=ir['right'];assert answer==('<' if a<b else '>' if a>b else '=')
        if stage==5:
            assert 100<=ir['left']<=9999 and 100<=ir['right']<=9999
            assert int(answer)==(ir['left']+ir['right'] if op=='add' else ir['left']-ir['right'])
        checked.add((stage,criterion))
    assert (5,'EXT-MULTI-DIGIT-SUBTRACT') in checked
