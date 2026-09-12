import pytest
from cftn_v3.full_curriculum_data import make
from cftn_v3.generated_correction import verify, integer_expression, correction_rows

def test_detects_first_borrow_error():
    row=make({'op':'subtract','left':18,'right':9,'detail':'place_value_v1'},1,'1NF-1')
    result=verify(row,'<work>8-0+0-9=9;1-0+0-0=1;9*1+1*10=19</work><answer>19</answer>')
    assert result['needs_correction'] and result['step_index']==0
    assert verify(row,row['target'])['kind']=='verified_canonical'
    assert verify(row,'<work>18-9=9</work><answer>9</answer>')['kind']=='verified_alternative'

def test_false_comparison_and_irrelevant_true_trace():
    row=make({'op':'compare','left':10,'right':20},0,'cmp')
    assert verify(row,'<work>cmp(10,20)==</work><answer>=</answer>')['needs_correction']
    result=verify(row,'<work>1+1=2</work><answer><</answer>')
    assert result['kind']=='unverified_alternative' and not result['needs_correction']

def test_no_arbitrary_expression_execution():
    with pytest.raises(ValueError):integer_expression("__import__('os')")

def test_supervision_uses_gold_targets_and_preserves_replay():
    row=make({'op':'subtract','left':18,'right':9},1,'sub')
    prior=make({'op':'compare','left':10,'right':20},0,'cmp')
    rows=correction_rows([row],[row],[prior],100,1)
    assert sum(r['stage']==0 for r in rows)==20
    assert all(r['target']==row['target'] for r in rows if r['stage']==1)
