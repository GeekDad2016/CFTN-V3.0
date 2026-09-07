"""Immutable, fully audited derivative of the 15-stage V11 curriculum."""
import argparse
import collections
import json
import math
import random
from pathlib import Path
from .config import canonical,identity
from .data import file_hash
from .local_specialist import MathTokenizer
from .math_procedures import solve,normalize_answer

SCOPE=['Counting, comparison and number structure','Small integer addition and subtraction',
 'Place value and crossing ten','Addition and subtraction within 100','Times tables and exact division',
 'Multi-digit integer arithmetic','Fractions, percentages and rectangle area','Linear equations, integer powers and Pythagoras',
 'Rational quadratic roots, two-variable systems and sequences','Polynomial differentiation and definite integration',
 'Finite combinatorics, binomial probabilities and constant acceleration','Small determinants, systems and polynomial evaluation',
 'Modular arithmetic, permutations and finite expectations','Selected series and finite algebra computations',
 'Polynomial identities, counterexamples and Euclidean certificates']

def semantic(ir):
    x={k:v for k,v in ir.items() if k not in ('type','strategy')}
    return identity(x)

def make(ir,index,criterion,source=None):
    ir={'type':'math_problem_v1',**ir};answer,target=solve(ir)
    return {'ir':ir,'prompt':canonical(ir),'target':target,'answer':answer,'phase':str(index),
        'stage':index,'criterion':criterion,'semantic_id':semantic(ir),'source_record':source}

def extensions(seed=9307,count=2400):
    rng=random.Random(seed)
    for _ in range(count):
        a=rng.choice([1,2,3,4,-1,-2]);x,y=rng.sample(range(-20,21),2)
        yield 8,'EXT-QUADRATIC-GENERAL',{'op':'quadratic_general','a':a,'b':-a*(x+y),'c':a*x*y}
        c=[rng.randint(-12,12) for _ in range(rng.randint(3,5))];c[-1]=rng.choice([1,2,3,-1,-2,-3])
        yield 9,'EXT-POLY-DERIVATIVE',{'op':'differentiate_polynomial','coefficients':c}
        lo=rng.randint(-4,2);hi=lo+rng.randint(1,5)
        yield 9,'EXT-POLY-INTEGRAL',{'op':'definite_integral_polynomial','coefficients':c,'lower':lo,'upper':hi}
        n=rng.randint(2,18);q=rng.randint(3,10);p=rng.randint(1,q-1)
        yield 10,'EXT-BINOMIAL-RATIONAL',{'op':'binomial_probability','trials':n,'successes':rng.randint(0,n),'probability':[p,q]}
        matrix=[[rng.randint(-8,8) for _ in range(3)] for _ in range(3)]
        yield 11,'EXT-DET-3X3',{'op':'matrix_det_3x3','matrix':matrix}
        yield 11,'EXT-DERIVATIVE-VALUE',{'op':'polynomial_derivative_value','coefficients':c,'point':rng.randint(-5,5)}
        a,b=rng.randint(2,200),rng.randint(2,200)
        yield 12,'EXT-BEZOUT',{'op':'bezout_coefficients','value':a,'modulus':b}
        weights=[rng.randint(1,8) for _ in range(3)];total=sum(weights)
        yield 12,'EXT-WEIGHTED-EXPECTATION',{'op':'weighted_expectation','values':rng.sample(range(-12,21),3),'weights':[[w,total] for w in weights]}
        q=rng.randint(1,12);p=rng.choice([i for i in range(-12,13) if i]);t=rng.randint(-15,15)
        yield 13,'EXT-SERIES-POINT',{'op':'geometric_series_point','ratio':[p,q],'point':[t,rng.randint(1,12)]}
        perm=list(range(rng.randint(4,8)));rng.shuffle(perm)
        yield 13,'EXT-PERMUTATION-ORDER',{'op':'permutation_order','permutation':perm}
        # Integer similarity transform of a diagonal matrix, including nontriangular matrices.
        x,y=rng.sample(range(-15,16),2);s=rng.choice([-3,-2,-1,1,2,3])
        yield 13,'EXT-GENERAL-SPECTRUM',{'op':'eigenvalues_2x2','matrix':[[x-s*(y-x),s*(y-x)],[-(s+1)*(y-x),y+s*(y-x)]]}
        right=list(c)
        if rng.randrange(2):right[rng.randrange(len(right))]+=rng.choice([-2,-1,1,2])
        else:right+=[0]
        yield 14,'EXT-POLY-IDENTITY',{'op':'polynomial_identity','left':c,'right':right}

def verify_manifest(destination):
    destination=Path(destination);m=json.loads((destination/'manifest.json').read_text())
    for name,digest in m['files'].items():
        if file_hash(destination/name)!=digest:raise ValueError('Dataset checksum changed: '+name)
    return m

def build(source,destination):
    source,destination=Path(source),Path(destination)
    if (destination/'manifest.json').exists():return verify_manifest(destination)
    destination.mkdir(parents=True,exist_ok=True)
    m=json.loads((source/'manifest.json').read_text());groups={s:[] for s in ('train','validation','test')}
    assignments={};old_assignments={};corrected={};seen=set();source_counts={}
    for split in groups:
        meta=m['splits'][split];path=source/meta['path']
        if file_hash(path)!=meta['sha256']:raise ValueError('Source hash mismatch')
        corrected[split]=0
        for line in path.open(encoding='utf-8'):
            raw=json.loads(line);r=make(raw['math_ir'],raw['curriculum_phase_index'],raw['criterion_id'],raw['record_id'])
            if normalize_answer(raw['answer'],r['ir']['op'])!=r['answer']:raise ValueError('Original answer disagrees with solver')
            corrected[split]+=raw['target_trace'].count('<answer>')!=1
            sid=r['semantic_id'];old=raw['math_semantic_id']
            if sid in assignments and assignments[sid]!=split:raise ValueError('Canonical input split leakage')
            if old in old_assignments and old_assignments[old]!=split:raise ValueError('Source semantic split leakage')
            assignments[sid]=split;old_assignments[old]=split;seen.add(sid);groups[split].append(r)
        source_counts[split]=len(groups[split])
        if len(groups[split])!=meta['records']:raise ValueError('Source count mismatch')
    extra=collections.Counter()
    for index,criterion,ir in extensions():
        r=make(ir,index,criterion);sid=r['semantic_id']
        if sid in seen:continue
        seen.add(sid)
        # Counterfactual identities share a split regardless of truth label.
        group=identity(['identity-family',ir['left']]) if ir['op']=='polynomial_identity' else sid
        bucket=int(group[:12],16)%10;split='train' if bucket<8 else 'validation' if bucket==8 else 'test'
        groups[split].append(r);extra[split]+=1
    files={};counts={};tokenizer=MathTokenizer();maximum=0;duplicates={}
    for split,rows in groups.items():
        path=destination/(split+'.jsonl');counts[split]=dict(collections.Counter(r['criterion'] for r in rows))
        duplicates[split]=len(rows)-len({r['semantic_id'] for r in rows})
        with path.open('w',encoding='utf-8',newline='\n') as out:
            for r in rows:
                length=len(tokenizer.prefix(r['prompt']))+len(tokenizer.encode(r['target']))+1
                maximum=max(maximum,length)
                if length>2048:raise ValueError(f'Context overflow: {length} {r["prompt"]}')
                if r['target'].count('<answer>')!=1:raise ValueError('Ambiguous answer')
                out.write(canonical(r)+'\n')
        files[path.name]=file_hash(path)
    # Remediation pools are training-only; shortest procedures first for scaffolded repair.
    stages=[]
    for i,phase in enumerate(m['phases']):
        rows=[r for r in groups['train'] if r['stage']==i];by=collections.defaultdict(list)
        for r in rows:by[r['criterion']].append(r)
        remediation=[]
        for criterion,items in by.items():
            unique={r['semantic_id']:r for r in items}
            remediation.extend(sorted(unique.values(),key=lambda r:(len(r['target']),r['semantic_id']))[:512])
        path=destination/f'remediation_{i:02d}.jsonl'
        path.write_text(''.join(canonical(r)+'\n' for r in remediation),encoding='utf-8');files[path.name]=file_hash(path)
        stages.append({'index':i,'name':phase['name'],'scope':SCOPE[i],'criteria':sorted(by),'remediation':path.name,
            'train_records':len(rows),'validation_records':sum(r['stage']==i for r in groups['validation'])})
    result={'format':'full_math_curriculum_v2','source':str(source),'source_manifest_sha256':file_hash(source/'manifest.json'),
        'source_records':source_counts,'corrected_multiple_answer_records':corrected,'added_records':dict(extra),
        'records':{s:len(v) for s,v in groups.items()},'files':files,'counts':counts,'stages':stages,
        'max_training_tokens':maximum,'duplicate_semantic_records':duplicates,
        'scope':'Exact bounded mathematical procedures. Not comprehensive graduate mathematics or research-level proof capability.',
        'evaluation_policy':'Fixed criterion-balanced validation; complete validation before promotion; sealed test only after all stages. No future-stage replay.'}
    (destination/'manifest.json').write_text(canonical(result),encoding='utf-8');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    m=build(a.source,a.output);print(json.dumps({k:v for k,v in m.items() if k not in ('files','counts','stages')},indent=2))
