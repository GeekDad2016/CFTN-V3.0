import random
from cftn_v3.arithmetic_scaffold import solve_scaffold
from cftn_v3.stage_first_repair import StageFirstController

def test_scaffolds_have_correct_intermediate_equations():
    rng=random.Random(8)
    for _ in range(100):
        a,b=rng.randint(100,9999),rng.randint(2,99)
        for ir,expected in [({'op':'add','left':a,'right':b},a+b),({'op':'subtract','left':a,'right':b},a-b),({'op':'multiply','left':a,'right':b},a*b),({'op':'divide','dividend':a*b,'divisor':b},a)]:
            answer,target=solve_scaffold(ir)
            assert int(answer)==expected
            for equation in target.split('</work>')[0].replace('<work>','').split(';'):
                lhs,rhs=equation.split('=')
                assert set(lhs)<=set('0123456789+-*/%')
                assert eval(lhs,{'__builtins__':{}})==int(rhs)

def test_recovery_requires_two_full_checks_and_then_normal_training():
    c=StageFirstController();c.state.update(mode='repair',focus='a',recovery_queue=['a','b'],repair_done=0,validation_enabled=True)
    for _ in range(30):c.observe([],[])
    assert c.state['mode']=='repair'
    c.recovery_result([]);assert c.state['focus']=='a'
    c.recovery_result(['a']);assert c.state['focus']=='a'
    c.recovery_result([]);c.recovery_result([]);assert c.state['focus']=='b'
    c.recovery_result([]);c.recovery_result([])
    assert c.state['mode']=='normal' and c.state['normal_done']==0
    assert c.state['validation_enabled']
