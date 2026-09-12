"""Explicit Stage 2 subskills, with balanced decisions and prior-stage replay."""
from .criterion_sampling import BalancedSampler, comparison_values

FOCI = ('expression_decision', 'borrowing', 'repeated_digits')

def matches(row, focus):
    ir = row['ir']
    if focus == 'borrowing':
        return (ir['op'] == 'subtract' and ir.get('detail') == 'place_value_v1'
                and ir['left'] % 10 < ir['right'] % 10)
    if ir['op'] != 'compare_expressions':
        return False
    if focus == 'expression_decision':
        return True
    return focus == 'repeated_digits' and 11 in comparison_values(ir)

def bucket(row):
    ir = row['ir']
    if ir['op'] != 'compare_expressions':
        return 'borrow'
    a, b = comparison_values(ir)
    if a == b: pattern = 'equal'
    elif a % 10 == b % 10: pattern = 'shared_units'
    elif len(str(a)) != len(str(b)): pattern = 'different_length'
    elif abs(a-b) == 1: pattern = 'adjacent'
    else: pattern = 'other'
    return pattern + ('/eleven' if 11 in (a,b) else '/ordinary')

def sample(active, prior, focus, count, seed):
    pool = [dict(r, criterion=r['criterion']+'/'+bucket(r)) for r in active if matches(r, focus)]
    if not pool: raise ValueError('Empty subskill training pool: '+focus)
    # 60% focused practice, 20% normal stage, 20% prior-stage retention.
    n = count*3//5; p = count//5 if prior else 0
    selected = BalancedSampler(pool).sample(n, seed)
    for r in selected: r['criterion'] = r['criterion'].split('/')[0]
    return selected + BalancedSampler(active).sample(count-n-p,seed+1) + BalancedSampler(prior).sample(p,seed+2)

def failed_subskills(report, rows):
    assert len(report['samples']) == len(rows)
    return [focus for focus in FOCI if any(matches(r,focus) and
            not all(s[k] for k in ('answer_correct','trace_correct','format_correct'))
            for r,s in zip(rows,report['samples']))]
