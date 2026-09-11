import json,shutil,collections
from pathlib import Path
from .full_curriculum_data import make,verify_manifest,semantic
from .equality_repair import pair
from .data import file_hash
from .config import canonical
from .math_procedures import score
from .local_specialist import MathTokenizer

def build(source,dest):
    source,dest=Path(source),Path(dest)
    assert not dest.exists()
    m=verify_manifest(source);groups={s:list(map(json.loads,(source/(s+'.jsonl')).open())) for s in ('train','validation','test')}
    seen={r['semantic_id'] for rs in groups.values() for r in rs}
    heldpairs={pair(r['ir']) for s in ('validation','test') for r in groups[s] if pair(r['ir']) is not None}
    q=source/'quarantined_heldout.jsonl'
    if q.exists():
        for r in map(json.loads,q.open()):
            if 'ir' in r:seen.add(semantic(r['ir']));heldpairs.add(pair(r['ir']))
    added=[];tok=MathTokenizer();maximum=0
    def add(ir,c):
        nonlocal maximum
        r=make(ir,0,c,'v3.1_foundation_expansion')
        if r['semantic_id'] in seen or (pair(ir) is not None and pair(ir) in heldpairs):return
        seen.add(r['semantic_id']);assert all(score(r['target'],r).values())
        size=len(tok.prefix(r['prompt']))+len(tok.encode(r['target']))+1;assert size<=2048;maximum=max(maximum,size);added.append(r)
    # Exhaust remaining legal small-number pairs rather than diluting them with large values.
    for a in range(101):
        for b in range(101):add({'op':'compare','left':a,'right':b},'1NPV-2')
    for a in range(101,1001):
        for b in (a,max(0,a-1),a+1):add({'op':'compare','left':a,'right':b},'1NPV-2')
    for a in range(1001):
        add({'op':'successor','value':a},'1NPV-1')
        if a:add({'op':'predecessor','value':a},'1NPV-1')
        for index in (1,2,3):
            seq=list(range(a,a+5));seq[index]=None
            add({'op':'missing_count_sequence','sequence':seq,'missing_index':index},'1NPV-1')
    shutil.copytree(source,dest);groups['train']+=added
    (dest/'train.jsonl').write_text(''.join(canonical(r)+'\n' for r in groups['train']),encoding='utf-8')
    stage=[r for r in groups['train'] if r['stage']==0]
    (dest/'remediation_00.jsonl').write_text(''.join(canonical(r)+'\n' for r in stage),encoding='utf-8')
    m['parent_manifest_sha256']=file_hash(source/'manifest.json');m['records']['train']=len(groups['train'])
    m['stages'][0].update(train_records=len(stage),unique_train_tasks=len({r['semantic_id'] for r in stage}),scope='Small sums, counting through 1,000 and range-balanced numeric comparisons')
    m['files']={name:file_hash(dest/name) for name in m['files']};(dest/'manifest.json').write_text(canonical(m))
    audit={'added':dict(collections.Counter(r['criterion'] for r in added)),'stage_records':len(stage),'max_new_tokens':maximum,'validation_and_test_unchanged':True}
    (dest/'foundation_audit.json').write_text(canonical(audit));print(json.dumps(audit))
if __name__=='__main__':build('G:/ctfn-text/data/v3_1_equality_v2','G:/ctfn-text/data/v3_1_foundation_v3')
