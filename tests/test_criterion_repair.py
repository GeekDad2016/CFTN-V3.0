import collections
import copy
from cftn_v3.criterion_sampling import BalancedSampler,balanced_panel
from cftn_v3.criterion_repair import RepairController,failures,add_strata

def row(i,label,criterion='compare',stage=1):
    a,b={'<':(1,2),'=':(2,2),'>':(3,2)}[label]
    return dict(criterion=criterion,stage=stage,semantic_id=str(i),split_group_id=str(i),answer=label,ir={'op':'compare','left':a,'right':b})

def test_balances_rare_equality_and_panel_without_duplicates():
    rows=[row(i,'<') for i in range(100)]+[row(101,'=')]+[row(i,'>') for i in range(102,202)]
    assert collections.Counter(r['answer'] for r in BalancedSampler(rows).sample(300,1))=={'<':100,'=':100,'>':100}
    panel=balanced_panel(rows,12)
    assert len(panel)==12 and len({r['semantic_id'] for r in panel})==12
    assert '=' in [r['answer'] for r in panel]

def test_repair_is_only_one_criterion_then_three_normal_rounds():
    c=RepairController();c.focus(['compare','other'],[])
    active=[row(1,'=') ,row(2,'<','other')];prior=[row(3,'>','previous',0)]
    assert {r['criterion'] for r in c.rows(active,prior,100,4)}=={'compare'}
    assert not c.observe(['other'],[])
    assert not c.observe(['other'],[])
    assert c.state['mode']=='consolidate'
    mixture=c.rows(active,prior,100,4)
    assert sum(r['stage']==0 for r in mixture)==25
    assert not c.observe([],[])
    assert not c.observe([],[])
    assert c.observe([],[])

def test_regression_after_consolidation_targets_other_and_resume_is_identical():
    c=RepairController();c.focus(['compare'],[]);c.observe([],[]);c.observe([],[])
    c.observe(['other'],[]);c.observe(['other'],[])
    resumed=RepairController(state=copy.deepcopy(c.state))
    assert not c.observe(['other'],[]) and not resumed.observe(['other'],[])
    assert c.state==resumed.state and c.state['focus']=='other'

def test_attempts_are_bounded_and_retention_can_be_focused():
    import pytest
    c=RepairController(repair=1,attempts=1);c.focus([],['previous'])
    with pytest.raises(RuntimeError,match='budget'):c.observe([],['previous'])

def test_per_class_failure_blocks_high_overall():
    m=dict(examples=100,accuracy=.99,trace_accuracy=.99,format_accuracy=1.)
    report={'criteria':{'compare':{**m,'strata':{'=':{**m,'accuracy':.5}}}}}
    assert failures(report)==['compare']

def test_new_stage_enters_targeted_repair_after_three_rounds():
    c=RepairController()
    for _ in range(3):assert not c.observe(['compare'],[])
    assert c.state['mode']=='repair'

def test_contrastive_generation_excludes_heldout_families():
    from cftn_v3.balanced_curriculum_data import contrastive_rows,comparison_family
    from cftn_v3.full_curriculum_data import make
    from cftn_v3.math_procedures import score
    held=[make({'op':'compare_expressions','left':[15,0],'right':[12,3],'left_op':'add','right_op':'add'},1,'EXT-COMPARE-EXPRESSIONS')]
    rows=contrastive_rows([],held,quota=2)
    assert collections.Counter(r['answer'] for r in rows)=={'<':16,'=':16,'>':16}
    assert not {comparison_family(r['ir']) for r in rows}&{comparison_family(r['ir']) for r in held}
    assert all(all(score(r['target'],r).values()) for r in rows)

def test_runner_repair_consolidation_and_promotion(tmp_path,monkeypatch):
    import contextlib,json
    from types import SimpleNamespace
    import cftn_v3.criterion_curriculum_training as t
    from cftn_v3.local_specialist import LocalMathTower
    from cftn_v3.full_curriculum_data import make
    spec={'hidden_size':8,'layers':1,'attention_heads':2,'feed_forward_size':16,'dropout':0.,'max_sequence_length':128}
    model=LocalMathTower(spec);monkeypatch.setattr(model,'to',lambda *a,**k:model)
    monkeypatch.setattr(t,'load_specialist',lambda p:(model,{'tower':'math','metadata':{}}))
    monkeypatch.setattr(t.torch.cuda,'is_available',lambda:True);monkeypatch.setattr(t.torch.cuda,'get_device_name',lambda:'mock')
    monkeypatch.setattr(t.torch.cuda,'memory_allocated',lambda:0);monkeypatch.setattr(t.torch,'autocast',lambda *a,**k:contextlib.nullcontext())
    updates=[]
    def loss(m,rows):updates.append(rows);return m.token_embedding.weight.square().mean()
    monkeypatch.setattr(t,'batch_loss',loss)
    r=make({'op':'add','left':2,'right':3},0,'addition')
    manifest={'stages':[{'name':'add','index':0,'scope':'add','remediation':'repair.jsonl'}]}
    monkeypatch.setattr(t,'verify_manifest',lambda p:manifest);monkeypatch.setattr(t,'file_hash',lambda p:'hash');monkeypatch.setattr(t,'read',lambda p:[r])
    def evaluate(m,rows,progress=None):
        accuracy=1. if len(updates)>=4 or not rows else 0.
        metric={'accuracy':accuracy,'format_accuracy':accuracy,'trace_accuracy':accuracy,'examples':len(rows)}
        return {**metric,'criteria':{'addition':metric} if rows else {},'samples':[]}
    monkeypatch.setattr(t,'evaluate',evaluate);saves=[]
    monkeypatch.setattr(t,'save_specialist',lambda p,m,meta,optimizer=None:saves.append(copy.deepcopy(meta)))
    args=SimpleNamespace(output=str(tmp_path/'run'),data='unused',initial_checkpoint='unused',normal_rounds=3,remediation_rounds=10,attempts=1,examples=4,lr=.001,consolidation_rounds=3,inherit_progress=False)
    t.run(args)
    reports=[json.loads(p.read_text()) for p in sorted((tmp_path/'run').glob('*_epoch_*.json'))]
    assert [r['training_mode'] for r in reports]==['normal']*3+['repair']*2+['consolidate']*3
    assert saves[-1]['accepted'] and saves[-1]['completed']==['add']
    assert not (tmp_path/'native_training.lock').exists()

def test_dashboard_uses_decision_gates(tmp_path):
    import json
    from cftn_v3.local_math_dashboard import snapshot
    m=dict(examples=100,accuracy=.99,trace_accuracy=1.,format_accuracy=1.,strata={'=':dict(accuracy=.5,trace_accuracy=1.,format_accuracy=1.)})
    (tmp_path/'status.json').write_text(json.dumps({'phase':'compare','state':'blocked'}))
    (tmp_path/'compare_epoch_001.json').write_text(json.dumps({'phase':'compare','epoch':1,'active':{'criteria':{'comparison':m}},'retention':{}}))
    assert not snapshot(tmp_path)['criterion_details'][0]['passed']
