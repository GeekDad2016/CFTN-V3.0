"""Stage-local validation scheduling; loss excludes the SIGReg penalty."""
import math

def validation_pending(state, policy):
    if state['mode'] != 'normal' or state.get('validation_enabled', False):
        return False
    if policy.get('validation_loss_threshold', 0):
        return True
    return state.get('normal_total', state['normal_done']) < policy['validation_warmup_rounds']

def unlock_validation(state, policy, mean_loss):
    reached = mean_loss is not None and math.isfinite(mean_loss) and mean_loss <= policy['validation_loss_threshold']
    exhausted = state['normal_done'] + 1 >= policy['normal_rounds']
    if reached or exhausted:
        state['validation_enabled'] = True
        state['validation_unlock_reason'] = 'Training-loss threshold reached' if reached else 'Normal-round budget reached; evaluating for recovery'
        return True
    return False
