from cftn_v3.stage_first_repair import StageFirstController

def test_normal_budget_precedes_repair_despite_persistent_failures():
    c=StageFirstController(normal=240)
    for r in range(1,240):
        c.observe(['subtraction'],[])
        if r%20==0:assert not c.full_result(r,['subtraction'],[])
        assert c.state['mode']=='normal'
    c.observe(['subtraction'],[])
    assert c.due(240,20,1320)
    assert not c.full_result(240,['subtraction'],[])
    assert c.state['mode']=='repair'

def test_repair_is_bounded_and_returns_to_full_normal_block():
    c=StageFirstController(normal=240,repair=30);c.focus(['subtraction'],[])
    for _ in range(30):c.observe(['subtraction'],[])
    assert c.state['mode']=='normal' and c.state['normal_done']==0
    assert c.state['recovery_total']==30
    for _ in range(10):c.observe(['subtraction'],[])
    c.full_result(280,['subtraction'],[])
    assert c.state['mode']=='normal'

def test_early_promotion_requires_two_full_checks_on_normal_rounds():
    c=StageFirstController(normal=240)
    for _ in range(3):c.observe([],[])
    assert c.due(3,20,1320)
    assert not c.full_result(3,[],[])
    c.observe([],[]);assert c.full_result(4,[],[])

def test_repair_retains_other_skills():
    def row(i,c):return {'criterion':c,'semantic_id':str(i),'answer':'3','ir':{'op':'add','left':1,'right':2}}
    c=StageFirstController();c.focus(['target'],[])
    rows=c.rows([row(1,'target'),row(2,'other')],[row(3,'prior')],100,1)
    assert sum(r['criterion']=='target' for r in rows)==80
    assert sum(r['criterion']!='target' for r in rows)==20
