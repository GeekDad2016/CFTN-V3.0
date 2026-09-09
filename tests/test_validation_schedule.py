from cftn_v3.validation_schedule import validation_pending, unlock_validation

def test_loss_gate_latches_and_resets_for_new_stage():
    policy = dict(validation_loss_threshold=.0035, normal_rounds=240)
    state = dict(mode='normal', normal_done=54)
    assert validation_pending(state, policy)
    for loss in (.004, None, float('nan')):
        assert not unlock_validation(state, policy, loss)
    assert unlock_validation(state, policy, .0035)
    assert not validation_pending(state, policy)
    state.update(normal_done=0, mode='repair')
    assert not validation_pending(state, policy)
    state['mode']='normal'
    assert not validation_pending(state, policy)
    assert validation_pending(dict(mode='normal', normal_done=0), policy)

def test_budget_forces_evaluation_without_claiming_threshold_reached():
    policy = dict(validation_loss_threshold=.0035, normal_rounds=240)
    state = dict(mode='normal', normal_done=239)
    assert unlock_validation(state, policy, .1)
    assert 'budget' in state['validation_unlock_reason']


def test_unbounded_warmup_ignores_all_round_limits():
    policy = dict(validation_loss_threshold=.0035, normal_rounds=240, unbounded_loss_warmup=1)
    state = dict(mode='normal', normal_done=10000, normal_total=10000)
    assert validation_pending(state, policy)
    assert not unlock_validation(state, policy, .004)
    assert unlock_validation(state, policy, .0035)
    assert state['normal_done']==0
    assert state['normal_total']==10000
    assert not validation_pending(state, policy)
