"""Balance meaningful decision classes, operations and procedural cases."""
import collections
import math
import random
from fractions import Fraction

COMPARISONS={'compare','compare_expressions','compare_place_value'}
BOOLEAN={'polynomial_identity','geometric_series_point'}

def decision(row):
    op=row['ir']['op']
    if op in COMPARISONS or op in BOOLEAN or op=='contains':return row['answer']
    return None

def comparison_values(ir):
    if ir['op']=='compare':return ir['left'],ir['right']
    if ir['op']=='compare_place_value':return ir['tens']*10+ir['ones'],ir['right']
    values=[]
    for side in ('left','right'):
        a,b=ir[side];values.append(a+b if ir[side+'_op']=='add' else a-b)
    return tuple(values)

def case(row):
    ir=row['ir'];op=ir['op']
    if op in COMPARISONS:
        a,b=comparison_values(ir);gap=abs(a-b)
        pair=ir.get('left_op','')+'/'+ir.get('right_op','')
        return pair+('/equal' if gap==0 else '/adjacent' if gap==1 else '/separated')
    if op=='add':return 'terms_'+str(len(ir.get('operands',[ir.get('left'),ir.get('right')])))
    if op=='power':return 'nonnegative_base' if ir['base']>=0 else 'negative_even' if ir['exponent']%2==0 else 'negative_odd'
    if op=='count':return 'zero' if row['answer']=='0' else 'nonzero'
    return 'standard'

def strata(row):return (row['ir']['op'],decision(row) or 'procedure',case(row))

class BalancedSampler:
    def __init__(self,rows):
        self.tree={}
        # Preserve different input representations, but not duplicate records.
        unique={(r['criterion'],r['semantic_id']):r for r in rows}
        for r in unique.values():
            branch=self.tree.setdefault(r['criterion'],{})
            for key in strata(r):branch=branch.setdefault(key,{})
            branch[r['semantic_id']]=r
        for operations in self.tree.values():
            for labels in operations.values():
                for cases in labels.values():
                    for name,values in cases.items():cases[name]=list(values.values())

    def sample(self,count,seed):
        rng=random.Random(seed);counters={};orders={};rows=[]
        if not self.tree:return rows
        def choose(branch,path):
            if path not in orders:
                keys=sorted(branch);rng.shuffle(keys);orders[path]=keys;counters[path]=0
            keys=orders[path];key=keys[counters[path]%len(keys)];counters[path]+=1
            return key,branch[key]
        for _ in range(count):
            branch=self.tree;path=()
            for _ in range(4):key,branch=choose(branch,path);path=path+(key,)
            rows.append(rng.choice(branch))
        rng.shuffle(rows);return rows

def balanced_panel(rows,per_criterion):
    groups=collections.defaultdict(dict)
    for r in rows:
        key=r.get('split_group_id',r['semantic_id']);groups[r['criterion']][key]=r
    result=[]
    for criterion,unique in sorted(groups.items()):
        buckets=collections.defaultdict(list)
        for r in sorted(unique.values(),key=lambda r:r['semantic_id']):buckets[strata(r)].append(r)
        # Round-robin by decision before procedural subcase; no replacement.
        tree={}
        for key,items in buckets.items():tree.setdefault(key[:2],[]).append(items)
        while len([r for r in result if r['criterion']==criterion])<per_criterion and tree:
            for primary in sorted(list(tree)):
                lists=tree[primary];items=lists.pop(0);result.append(items.pop(0))
                if items:lists.append(items)
                if not lists:del tree[primary]
                if len([r for r in result if r['criterion']==criterion])>=per_criterion:break
    return result

def audit(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['criterion']].append(r)
    result={}
    for name,items in sorted(groups.items()):
        unique=list({r['semantic_id']:r for r in items}.values())
        labels=collections.Counter(decision(r) for r in unique if decision(r) is not None)
        operations=collections.Counter(r['ir']['op'] for r in unique);cases=collections.Counter(case(r) for r in unique)
        warnings=[];expected=set()
        if any(op in COMPARISONS for op in operations):expected={'<','=','>'}
        elif any(op in BOOLEAN for op in operations):expected={'true','false'}
        elif 'contains' in operations:expected={'yes','no'}
        if expected-set(labels):warnings.append('missing_decision_class')
        if labels and min(labels.values())/sum(labels.values())<.15:warnings.append('rare_decision_class')
        if len(operations)>1 and max(operations.values())>2*min(operations.values()):warnings.append('operation_imbalance')
        answers=collections.Counter(r['answer'] for r in unique)
        if len(answers)==1:warnings.append('constant_answer_contract_requires_procedure_checks')
        numeric=collections.Counter()
        for r in unique:
            try:v=Fraction(r['answer']);numeric['zero' if v==0 else 'negative' if v<0 else 'positive']+=1
            except (ValueError,ZeroDivisionError):pass
        result[name]={'records':len(items),'unique_inputs':len(unique),'decision_counts':dict(labels),
            'operations':dict(operations),'procedural_cases':dict(cases),'numeric_sign_counts':dict(numeric),'warnings':warnings,
            'sampling_policy':'Uniform criteria, operations, decision classes and procedural cases; uniform examples within each case. Numeric answer values are not treated as separate classes.'}
    return result
