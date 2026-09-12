"""Training-only generated-error mining across the maths curriculum."""
import math
import random
from .balanced_curriculum_data import comparison_family
from .criterion_sampling import balanced_panel, BalancedSampler
from .generated_correction import verify, correction_rows
from .math_procedures import solve, score

def check(row, output, complete=True):
    answer,target=solve(row['ir'])
    if (answer,target)!=(row['answer'],row['target']):raise ValueError('Dataset target disagrees with procedure solver')
    metrics=score(output,row,complete)
    if not metrics['answer_correct'] or not metrics['format_correct']:
        return {'needs_correction':True,'kind':'wrong_answer_or_format',**metrics}
    if metrics['trace_correct']:return {'needs_correction':False,'kind':'canonical',**metrics}
    # Only diagnose step arithmetic in the bounded verifier's supported domain.
    try:feedback=verify(row,output)
    except (ValueError,KeyError):
        feedback={'needs_correction':False,'kind':'unverified_alternative'}
    return {**feedback,**metrics}

def mining_panel(active,prior,reserved,seed,limit=256):
    def select(rows,count):
        rows=[r for r in rows if comparison_family(r['ir']) not in reserved]
        criteria={r['criterion'] for r in rows}
        panel=balanced_panel(rows,math.ceil(count/max(1,len(criteria))))
        random.Random(seed).shuffle(panel)
        return panel[:count]
    return select(active,limit*3//4 if prior else limit)+select(prior,limit//4 if prior else 0)

def due(state,round_,warmup):
    return not warmup and state['mode']=='normal' and round_-state.get('correction_mined_round',-1000)>=20

def rows(state,active,prior,count,seed):
    lookup={r['semantic_id']:r for r in active+prior}
    ids=state.get('correction_ids',[])
    if any(i not in lookup for i in ids):raise ValueError('Correction pool is not part of training data')
    failed=[lookup[i] for i in ids]
    active_ids={r['semantic_id'] for r in active}
    focus=[r for r in failed if r['semantic_id'] in active_ids]
    old=[r for r in failed if r['semantic_id'] not in active_ids]
    n=count*3//5 if focus else 0
    p=count//5 if prior else 0
    return (BalancedSampler(focus).sample(n,seed)
            +BalancedSampler(active).sample(count-n-p,seed+1)
            +BalancedSampler(old or prior).sample(p//2,seed+2)
            +BalancedSampler(prior).sample(p-p//2,seed+3))
