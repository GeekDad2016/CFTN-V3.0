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
