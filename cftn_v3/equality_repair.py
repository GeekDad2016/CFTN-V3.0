"""Immutable contrastive comparison expansion; existing held-out pairs stay held out."""
import json,random,shutil,collections
from pathlib import Path
from .full_curriculum_data import make,verify_manifest
from .config import canonical
from .data import file_hash
from .local_specialist import MathTokenizer
from .math_procedures import score

def pair(ir):
    if ir.get('op')!='compare':return None
    return tuple(sorted((ir['left'],ir['right'])))

def build(source,dest):
    source,dest=Path(source),Path(dest)
    if dest.exists():return verify_manifest(dest)
    m=verify_manifest(source);groups={s:[json.loads(x) for x in (source/(s+'.jsonl')).open()] for s in ('train','validation','test')}
    occupied={pair(r['ir']) for rows in groups.values() for r in rows if pair(r['ir']) is not None}
    qp=source/'quarantined_heldout.jsonl'
    if qp.exists():occupied.update(pair(r['ir']) for r in map(json.loads,qp.open()) if r.get('ir') and pair(r['ir']) is not None)
    rng=random.Random(311911);values=list(range(0,10000));rng.shuffle(values)
    extra={'train':[],'validation':[]};diagnostic=[]
    for split,n in [('validation',60),('train',3000)]:
        accepted=0
        for a in values:
            keys={(a,a),(a,a+1)}
            if keys&occupied:continue
            occupied.update(keys);accepted+=1
            for left,right in [(a,a),(a,a+1),(a+1,a)]:
                r=make({'op':'compare','left':left,'right':right},0,'1NPV-2','v3.1_equality_contrastive')
                extra[split].append(r)
                if split=='validation' and left==right and len(diagnostic)<30:diagnostic.append(r)
            if accepted==n:break
        assert accepted==n
    tok=MathTokenizer();maximum=0
    for rows in extra.values():
        for r in rows:
            assert all(score(r['target'],r).values())
            maximum=max(maximum,len(tok.prefix(r['prompt']))+len(tok.encode(r['target']))+1)
    assert maximum<=2048
    shutil.copytree(source,dest)
    for split,rs in extra.items():
        groups[split]+=rs
        (dest/(split+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in groups[split]),encoding='utf-8')
    stage=[r for r in groups['train'] if r['stage']==0]
    (dest/'remediation_00.jsonl').write_text(''.join(canonical(r)+'\n' for r in stage),encoding='utf-8')
    m['stages'][0].update(train_records=len(stage),unique_train_tasks=len({r['semantic_id'] for r in stage}),validation_records=sum(r['stage']==0 for r in groups['validation']),scope='Number structure and small sums, with contrastive comparisons through four digits')
    m['records']={s:len(rs) for s,rs in groups.items()};m['parent_manifest_sha256']=file_hash(source/'manifest.json');m['equality_release']='contrastive_v1'
    m['files']={name:file_hash(dest/name) for name in m['files']}
    (dest/'manifest.json').write_text(canonical(m),encoding='utf-8')
    (dest/'equality_diagnostic.json').write_text(canonical(diagnostic),encoding='utf-8')
    report={'added_train':len(extra['train']),'added_validation':len(extra['validation']),'fresh_equality_training_pairs':3000,'max_tokens':maximum,'original_splits_preserved':True}
    (dest/'equality_audit.json').write_text(canonical(report));print(json.dumps(report))
    return verify_manifest(dest)
if __name__=='__main__':build('G:/ctfn-text/data/v3_1_main','G:/ctfn-text/data/v3_1_equality_v2')
