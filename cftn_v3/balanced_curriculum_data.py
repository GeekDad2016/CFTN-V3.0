"""Preserve prior data and add audited contrastive comparison supervision."""
import argparse
import collections
import json
import random
import shutil
from pathlib import Path
from .config import canonical,identity
from .data import file_hash
from .full_curriculum_data import make,verify_manifest
from .criterion_sampling import audit,case
from .local_specialist import MathTokenizer
from .math_procedures import score

def comparison_family(ir):
    op=ir['op']
    if op=='compare':return identity([op,*sorted([ir['left'],ir['right']])])
    if op!='compare_expressions':return identity({k:v for k,v in ir.items() if k not in ('type','strategy')})
    expressions=[]
    for side in ('left','right'):
        operator=ir[side+'_op'];values=sorted(ir[side]) if operator=='add' else ir[side]
        expressions.append([operator,values])
    return identity([op,*sorted(expressions,key=canonical)])

def contrastive_rows(existing,heldout,quota=96):
    rng=random.Random(9321);seen={r['semantic_id'] for r in existing+heldout}
    reserved={comparison_family(r['ir']) for r in heldout};added=[];counts=collections.Counter()
    keys=[(label,left,right,gap) for label in ('<','=','>') for left in ('add','subtract') for right in ('add','subtract')
        for gap in (('equal',) if label=='=' else ('adjacent','separated'))]
    def operands(value,op):
        if op=='add':a=rng.randint(0,value);return [a,value-a]
        b=rng.randint(0,20-value);return [value+b,b]
    def target(key):return quota*(2 if key[0]=="=" else 1)
    for key in keys:
        label,left,right,gap=key;tries=0
        while counts[key]<target(key) and tries<100000:
            tries+=1;distance=0 if gap=='equal' else 1 if gap=='adjacent' else rng.randint(2,20)
            a=rng.randint(0,20-distance);b=a+distance
            if label=='>':a,b=b,a
            ir={'op':'compare_expressions','left':operands(a,left),'right':operands(b,right),'left_op':left,'right_op':right}
            reverse={'op':ir['op'],'left':ir['right'],'right':ir['left'],'left_op':right,'right_op':left}
            for request in (ir,reverse):
                r=make(request,1,'EXT-COMPARE-EXPRESSIONS');family=comparison_family(request)
                this_key=(r['answer'],request['left_op'],request['right_op'],case(r).split('/')[-1])
                if counts[this_key]>=target(this_key) or r['semantic_id'] in seen or family in reserved:continue
                r['split_group_id']=r['semantic_id'];seen.add(r['semantic_id']);added.append(r);counts[this_key]+=1
        if counts[key]<target(key):raise ValueError('Insufficient novel comparison families for '+str(key))
    return added

def build(source,destination,tower):
    source,destination=Path(source),Path(destination)
    if (destination/'manifest.json').exists():return verify_manifest(destination)
    parent=verify_manifest(source);destination.mkdir(parents=True,exist_ok=True)
    groups={s:[json.loads(l) for l in (source/(s+'.jsonl')).open(encoding='utf-8')] for s in ('train','validation','test')}
    before={s:audit(rows) for s,rows in groups.items()};augment=[];new_validation=collections.Counter()
    if tower=='math':
        heldout=groups['validation']+groups['test']
        quarantine=source/'quarantined_heldout.jsonl'
        if quarantine.exists():heldout += [json.loads(l)['record'] for l in quarantine.open(encoding='utf-8')]
        augment=contrastive_rows(groups['train'],heldout);groups['train'].extend(augment)
        # Supplement missing/rare elementary comparison classes without reusing any old family.
        reserved={comparison_family(r['ir']) for rows in groups.values() for r in rows}
        reserved.update(comparison_family(r['ir']) for r in heldout)
        candidates=[make({'op':'compare','left':a,'right':b},0,'1NPV-2') for a in range(101) for b in range(101)]
        random.Random(9322).shuffle(candidates)
        for split in ('validation','test'):
            counts=collections.Counter(r['answer'] for r in groups[split] if r['criterion']=='1NPV-2')
            for r in candidates:
                family=comparison_family(r['ir'])
                if counts[r['answer']]>=12 or family in reserved:continue
                r={**r,'split_group_id':r['semantic_id']};groups[split].append(r);reserved.add(family)
                counts[r['answer']]+=1;new_validation[split]+=1
    files={};maximum=0;tok=MathTokenizer();limit=2048 if tower=='math' else 256
    for split,rows in groups.items():
        path=destination/(split+'.jsonl')
        with path.open('w',encoding='utf-8',newline='\n') as f:
            for r in rows:
                n=len(tok.prefix(r['prompt']))+len(tok.encode(r['target']))+1;maximum=max(maximum,n)
                if n>limit or not all(score(r['target'],r).values()):raise ValueError('Invalid supervision')
                f.write(canonical(r)+'\n')
        files[path.name]=file_hash(path)
    stages=[]
    for old in parent['stages']:
        stage=dict(old);rows=[r for r in groups['train'] if r['stage']==stage['index']]
        unique={(r['criterion'],r['semantic_id']):r for r in rows}
        path=destination/stage['remediation'];path.write_text(''.join(canonical(r)+'\n' for r in unique.values()),encoding='utf-8');files[path.name]=file_hash(path)
        stage.update(train_records=len(rows),validation_records=sum(r['stage']==stage['index'] for r in groups['validation']),
            unique_train_tasks=len({r.get('split_group_id',r['semantic_id']) for r in rows}))
        stages.append(stage)
    for name in ('initial.specialist','quarantined_heldout.jsonl'):
        if (source/name).exists():shutil.copyfile(source/name,destination/name);files[name]=file_hash(destination/name)
    report={'before':before,'after':{s:audit(rows) for s,rows in groups.items()},
        'contrastive_training_added':len(augment),'additional_heldout':dict(new_validation),
        'notes':['Decision outcomes, mixed operations and procedural cases are balanced by the sampler, not by deleting majority examples.',
            'Fixed true identities retain correct labels; their answer accuracy alone is uninformative. Numeric sign counts are diagnostics, not a requirement for uniform numeric answers.',
            'New comparison training excludes held-out reversed/commuted expression families. This does not assert arbitrary mathematical-equivalence isolation for inherited data.']}
    path=destination/'balance_audit.json';path.write_text(canonical(report),encoding='utf-8');files[path.name]=file_hash(path)
    keys={s:{r.get('split_group_id',r['semantic_id']) for r in rows} for s,rows in groups.items()}
    assert not(keys['train']&keys['validation'] or keys['train']&keys['test'] or keys['validation']&keys['test'])
    result={**parent,'format':'balanced_'+tower+'_curriculum_v4','parent':str(source),'parent_manifest_sha256':file_hash(source/'manifest.json'),
        'files':files,'stages':stages,'records':{s:len(rows) for s,rows in groups.items()},'max_training_tokens':maximum,
        'counts':{s:dict(collections.Counter(r['criterion'] for r in rows)) for s,rows in groups.items()},
        'duplicate_semantic_records':{s:len(rows)-len({r['semantic_id'] for r in rows}) for s,rows in groups.items()},
        'contrastive_training_added':len(augment),'additional_heldout':dict(new_validation),'balance_audit':'balance_audit.json'}
    (destination/'manifest.json').write_text(canonical(result),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--tower',choices=['math','string'],required=True);a=p.parse_args()
    m=build(a.source,a.output,a.tower);print({k:m[k] for k in ('records','contrastive_training_added','additional_heldout','max_training_tokens')})
