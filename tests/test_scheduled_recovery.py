from cftn_v3.criterion_repair import ScheduledRepairController

def test_no_attempt_limit_and_twenty_round_checks():
    c=ScheduledRepairController(attempts=1)
    for _ in range(100):c.focus(['comparison'],[])
    assert c.state['mode']=='repair'
    assert [i for i in range(47,81) if c.due(i,20,240)]==[60,80]

def test_full_success_requires_consecutive_rounds_and_normal_consolidation():
    c=ScheduledRepairController();c.state.update(mode='consolidate',focus='comparison',consolidation_done=3)
    assert not c.full_result(60,[],[])
    assert c.due(61,20,240)
    assert c.full_result(61,[],[])
    assert not c.full_result(80,[],[])
    assert not c.full_result(81,['comparison'],[])
    assert c.state['mode']=='repair' and c.state['full_streak']==0
    assert not c.due(82,20,240)

def test_quick_pass_does_not_establish_full_success_and_resume_preserves_followup():
    import copy
    c=ScheduledRepairController();c.focus(['comparison'],[])
    for _ in range(8):c.observe([],[])
    assert not c.state.get('full_streak')
    assert not c.full_result(60,[],[])
    resumed=ScheduledRepairController(state=copy.deepcopy(c.state))
    assert resumed.due(61,20,240) and resumed.full_result(61,[],[])
    assert resumed.due(240,20,240)

def test_normal_rounds_before_repair_do_not_replace_consolidation():
    c=ScheduledRepairController();c.state['normal_done']=3;c.focus(['comparison'],[])
    assert not c.full_result(60,[],[])
    assert not c.full_result(61,[],[])
    assert c.state['mode']=='consolidate'

def test_immediate_check_after_three_normal_rounds_then_next_round():
    c=ScheduledRepairController();c.focus(['comparison'],[])
    c.observe([],[]);c.observe([],[])
    assert c.next_check(123,20,240)==125
    c.observe([],[]);assert not c.due(123,20,240)
    c.observe([],[]);assert not c.due(124,20,240)
    assert c.next_check(125,20,240)==125
    c.observe([],[]);assert c.due(125,20,240)
    assert not c.full_result(125,[],[])
    c.observe([],[]);assert c.due(126,20,240)
    assert c.full_result(126,[],[])

def test_full_failure_returns_to_repair_and_twenty_round_fallback():
    c=ScheduledRepairController();c.state.update(mode='consolidate',consolidation_done=3)
    assert not c.full_result(125,['comparison'],[])
    assert not c.due(126,20,240)
    assert c.next_check(126,20,240)==140
