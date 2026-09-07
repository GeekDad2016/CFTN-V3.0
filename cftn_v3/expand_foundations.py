"""Versioned foundation expansion; preserve source training and quarantine leaked tests."""
import argparse
import collections
import itertools
import json
import random
from pathlib import Path
from .config import canonical,identity
from .data import file_hash
from .full_curriculum_data import make,verify_manifest
from .local_specialist import MathTokenizer
from .math_procedures import score

def group_id(ir):
    op=ir['op']
    if op=='add':return identity([op,*sorted(ir.get('operands',[ir.get('left'),ir.get('right')]))])
    if op=='multiply':return identity([op,*sorted([ir['left'],ir['right']])])
    return identity({k:v for k,v in ir.items() if k not in ('type','strategy')})

def candidates():
    # Finite exhaustive facts, not thousands of copies of the same sum.
    for size in (2,3):
        for values in itertools.product(range(21),repeat=size):
            if sum(values)<=20:yield 0,'1AS-1',{'op':'add','operands':list(values)}
    for a in range(21):
        for b in range(21):
            if a+b<=20:yield 1,'1NF-1',{'op':'add','left':a,'right':b}
            if a>=b:
                yield 1,'1NF-1',{'op':'subtract','left':a,'right':b}
                yield 1,'1AS-2',{'op':'missing_addend','total':a,'known':b}
                yield 1,'EXT-MISSING-SUBTRAHEND',{'op':'missing_subtrahend','left':a,'result':b}
                yield 1,'EXT-MISSING-MINUEND',{'op':'missing_minuend','result':b,'right':a-b}
    rng=random.Random(9311)
    for _ in range(6500):
        sides={}
        for side in ('left','right'):
            a,b=rng.randrange(21),rng.randrange(21);op=rng.choice(['add','subtract'])
            if op=='add':b=rng.randrange(21-a)
            else:a,b=max(a,b),min(a,b)
            sides[side]=[a,b];sides[side+'_op']=op
        yield 1,'EXT-COMPARE-EXPRESSIONS',{'op':'compare_expressions',**sides}
    for value in range(10,100):
        yield 2,'2NPV-1',{'op':'place_value','value':value}
        yield 2,'2NPV-2',{'op':'neighbouring_tens','value':value}
        yield 2,'EXT-COMPOSE-PLACE-VALUE',{'op':'compose_place_value','tens':value//10,'ones':value%10}
        for delta in range(1,10):
            if value+delta<100 and value//10!=(value+delta)//10:yield 2,'2AS-1',{'op':'add','left':value,'right':delta}
            if value-delta>=0 and value//10!=(value-delta)//10:yield 2,'2AS-1',{'op':'subtract','left':value,'right':delta}
        for known in range(max(0,value-19),value):
            yield 2,'EXT-TWO-DIGIT-MISSING',{'op':'missing_addend','total':value,'known':known}
        for right in range(10,100):
            yield 2,'EXT-COMPARE-PLACE-VALUE',{'op':'compare_place_value','tens':value//10,'ones':value%10,'right':right}
    # Add explicit carry/borrow tasks across hundreds and thousands, beyond the old 100..299 range.
    for digits in (3,4):
        low,high=10**(digits-1),10**digits-1
        for _ in range(4000):
            a,b=rng.randint(low,high),rng.randint(low,high)
            yield 5,'KS2-MULTI-DIGIT',{'op':'add','left':a,'right':b}
            yield 5,'EXT-MULTI-DIGIT-SUBTRACT',{'op':'subtract','left':max(a,b),'right':min(a,b)}

def build(source,destination):
    source,destination=Path(source),Path(destination)
    if (destination/'manifest.json').exists():return verify_manifest(destination)
    parent=verify_manifest(source);destination.mkdir(parents=True,exist_ok=True)
    groups={};owners={};seen=set();quarantine=[];old_counts={}
    for split in ('train','validation','test'):
        groups[split]=[]
        for line in (source/(split+'.jsonl')).open(encoding='utf-8'):
            r=json.loads(line);key=group_id(r['ir']);r['split_group_id']=key
            if key in owners and owners[key]!=split:
                # A pretrained training question cannot become a new clean held-out question.
                quarantine.append({'original_split':split,'reason':'Equivalent task already in '+owners[key],'record':r});continue
            owners[key]=split;seen.add((r['stage'],r['criterion'],r['semantic_id']));groups[split].append(r)
        old_counts[split]=len(groups[split])
    added=collections.Counter();reserved={s:{(r['split_group_id'],r['criterion']) for r in rows} for s,rows in groups.items()}
    for stage,criterion,ir in candidates():
        r=make(ir,stage,criterion);key=group_id(ir);r['split_group_id']=key
        record_key=(stage,criterion,r['semantic_id'])
        if record_key in seen:continue
        seen.add(record_key)
        if key in owners:split=owners[key]
        else:
            bucket=int(key[:12],16)%10;split='train' if bucket<8 else 'validation' if bucket==8 else 'test';owners[key]=split
        # Do not create extra variants of an already reserved evaluation task.
        if split!='train' and (key,criterion) in reserved[split]:continue
        reserved[split].add((key,criterion))
        groups[split].append(r);added[split]+=1
    files={};maximum=0;tok=MathTokenizer()
    for split,rows in groups.items():
        path=destination/(split+'.jsonl')
        with path.open('w',encoding='utf-8',newline='\n') as f:
            for r in rows:
                length=len(tok.prefix(r['prompt']))+len(tok.encode(r['target']))+1;maximum=max(maximum,length)
                if length>2048 or not all(score(r['target'],r).values()):raise ValueError('Invalid target or token overflow')
                f.write(canonical(r)+'\n')
        files[path.name]=file_hash(path)
    path=destination/'quarantined_heldout.jsonl';path.write_text(''.join(canonical(r)+'\n' for r in quarantine),encoding='utf-8');files[path.name]=file_hash(path)
    stages=[]
    scopes={0:'Counting, comparison and small sums up to 20',1:'Addition, subtraction and missing values within 20',
        2:'Two-digit place value and crossing tens',4:'Multiplication and exact division with factors up to 100',
        5:'Multi-digit arithmetic, including addition and subtraction with operands up to 9,999'}
    for old in parent['stages']:
        i=old['index'];stage={**old,'scope':scopes.get(i,old['scope'])};by=collections.defaultdict(list)
        for r in groups['train']:
            if r['stage']==i:by[r['criterion']].append(r)
        repair=[]
        for criterion,rows in by.items():
            # Cover both short and long procedures, rather than favouring only short traces.
            ordered=sorted({r['semantic_id']:r for r in rows}.values(),key=lambda r:(len(r['target']),r['semantic_id']))
            repair.extend(ordered if len(ordered)<=768 else ordered[:256]+random.Random(9311+i).sample(ordered[256:],512))
        path=destination/stage['remediation'];path.write_text(''.join(canonical(r)+'\n' for r in repair),encoding='utf-8');files[path.name]=file_hash(path)
        stage.update(criteria=sorted(by),train_records=sum(map(len,by.values())),
            unique_train_tasks=len({r['split_group_id'] for rows in by.values() for r in rows}),
            validation_records=sum(r['stage']==i for r in groups['validation']))
        stages.append(stage)
    result={**parent,'format':'full_math_curriculum_v3','parent':str(source),'parent_manifest_sha256':file_hash(source/'manifest.json'),
        'files':files,'stages':stages,'records':{s:len(rows) for s,rows in groups.items()},
        'counts':{s:dict(collections.Counter(r['criterion'] for r in rows)) for s,rows in groups.items()},
        'foundation_added_records':dict(added),'duplicate_semantic_records':{s:len(rows)-len({r['semantic_id'] for r in rows}) for s,rows in groups.items()},
        'quarantined_heldout_records':dict(collections.Counter(r['original_split'] for r in quarantine)),
        'max_training_tokens':maximum,'split_policy':'Equivalent addition forms and commuted addition/multiplication share a split; contaminated held-out records are quarantined, never treated as new clean tests.'}
    # Recheck group isolation, including newly added representations.
    keys={s:{r['split_group_id'] for r in rows} for s,rows in groups.items()}
    assert not(keys['train']&keys['validation'] or keys['train']&keys['test'] or keys['validation']&keys['test'])
    (destination/'manifest.json').write_text(canonical(result),encoding='utf-8');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    m=build(a.source,a.output);print(json.dumps({'records':m['records'],'quarantine':m['quarantined_heldout_records'],
        'stages':[{k:s[k] for k in ('index','scope','train_records','unique_train_tasks','validation_records')} for s in m['stages'][:6]]},indent=2))
