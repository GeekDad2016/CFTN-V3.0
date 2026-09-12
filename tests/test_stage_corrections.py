from cftn_v3.stage_corrections import due, check, mining_panel, rows
from cftn_v3.full_curriculum_data import make
from cftn_v3.balanced_curriculum_data import comparison_family

def test_no_mining_before_loss_gate_and_bounded_cadence():
    state={'mode':'normal'}
    assert not due(state,100,True)
    assert due(state,100,False)
    state['correction_mined_round']=100
    assert not due(state,119,False)
    assert due(state,120,False)
    state['mode']='repair'
    assert not due(state,140,False)

def test_later_stage_gold_and_wrong_answer():
    row=make({'op':'differentiate_polynomial','coefficients':[1,2,3]},9,'poly')
    assert not check(row,row['target'])['needs_correction']
    assert check(row,'<work>1+1=2</work><answer>0</answer>')['needs_correction']

def test_reserved_family_exclusion_and_prior_fraction():
    a=make({'op':'add','left':7,'right':9},1,'add')
    b=make({'op':'add','left':8,'right':9},1,'add')
    old=make({'op':'compare','left':24,'right':33},0,'cmp')
    panel=mining_panel([a,b],[old],{comparison_family(a['ir'])},1)
    assert a not in panel and b in panel
    batch=rows({'correction_ids':[old['semantic_id']]},[a,b],[old],100,1)
    assert sum(r['stage']==0 for r in batch)==20
