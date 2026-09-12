from cftn_v3.full_curriculum_data import make
from cftn_v3.subskill_recovery import matches, bucket, sample
from cftn_v3.stage_first_repair import StageFirstController

def test_borrow_and_comparison_patterns():
    borrow=make({'op':'subtract','left':18,'right':9,'detail':'place_value_v1'},1,'1NF-1')
    assert matches(borrow,'borrowing')
    row=make({'op':'compare_expressions','left':[11,1],'right':[16,4],
              'left_op':'subtract','right_op':'add'},1,'EXT-COMPARE-EXPRESSIONS')
    assert bucket(row)=='shared_units/ordinary'
    prior=make({'op':'compare','left':2,'right':3},0,'1NPV-2')
    rows=sample([row,borrow],[prior],'borrowing',100,42)
    assert sum(r['stage']==0 for r in rows)==20
    assert sum(matches(r,'borrowing') for r in rows)>=60
    assert borrow['criterion']=='1NF-1'

def test_small_passes_cannot_end_subskill_recovery():
    c=StageFirstController()
    c.state.update(subskill_recovery=True,pending_subskills=['borrowing'])
    c.focus(['1NF-1'],[])
    for _ in range(4): c.observe([],[])
    assert c.state['mode']=='repair' and c.state['recovery_queue']==['borrowing']
    c.recovery_result([])
    assert c.state['mode']=='repair'
    c.recovery_result(['borrowing'])
    assert c.state['recovery_gate_streak']==0
    c.recovery_result([]);c.recovery_result([])
    assert c.state['mode']=='normal' and c.state['normal_done']==0

def test_adoption_requires_improvement_and_retention():
    from cftn_v3.subskill_probe import choose
    def report(active_errors, retention_errors=0):
        def panel(n):return {'samples':[dict(answer_correct=i>=n,trace_correct=i>=n,format_correct=True) for i in range(10)]}
        return {'active':panel(active_errors),'retention':panel(retention_errors)}
    assert choose({'start':report(3),'baseline':report(2),'targeted':report(1)})=='targeted'
    assert choose({'start':report(3),'baseline':report(2),'targeted':report(1,1)})=='baseline'
    assert choose({'start':report(1),'baseline':report(2),'targeted':report(2)})=='start'
    assert choose({'start':report(3),'baseline':report(2),'targeted':report(2)})=='baseline'
