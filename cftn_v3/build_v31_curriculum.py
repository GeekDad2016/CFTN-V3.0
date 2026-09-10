"""Build V3.1 main curriculum with remediation and early arithmetic scaffolds."""
import json,shutil,collections
from pathlib import Path
from .full_curriculum_data import make,verify_manifest
from .build_arithmetic_recovery import family,subcase
from .math_procedures import score
from .local_specialist import MathTokenizer
from .config import canonical
from .data import file_hash

def eligible(ir):
    op=ir['op']
    if ir.get('detail'):return False
    if op in ('add','subtract','multiply'):
        a,b=ir.get('left'),ir.get('right')
        return type(a)==int and type(b)==int and a>=0 and b>=0 and (op!='subtract' or a>=b)
    if op=='divide':
        a,b=ir['dividend'],ir['divisor']
        return type(a)==int and type(b)==int and a>=0 and b>0 and a%b==0
    return False

def build(source,dest):
    source,dest=Path(source),Path(dest)
    if dest.exists():raise ValueError('Refusing to overwrite V3.1 dataset')
    manifest=verify_manifest(source)
    groups={s:[json.loads(x) for x in (source/(s+'.jsonl')).open()] for s in ('train','validation','test')}
    heldout={family(r['ir']) for s in ('validation','test') for r in groups[s]}
    quarantine=source/'quarantined_heldout.jsonl'
    if quarantine.exists():heldout.update(family(r['ir']) for r in map(json.loads,quarantine.open()) if 'ir' in r)
    seen={(r['criterion'],r['semantic_id']) for r in groups['train']};merged=0
    for path in sorted(source.glob('remediation_*.jsonl')):
        for line in path.open():
            r=json.loads(line);key=(r['criterion'],r['semantic_id'])
            if key not in seen and family(r['ir']) not in heldout:
                groups['train'].append(r);seen.add(key);merged+=1
    added=collections.Counter();skipped=0
    for split,rows in groups.items():
        existing={(r['criterion'],r['semantic_id']) for r in rows};extra=[]
        for r in rows:
            if not eligible(r['ir']):continue
            if split=='train' and family(r['ir']) in heldout:skipped+=1;continue
            new=make({**r['ir'],'detail':'place_value_v1'},r['stage'],r['criterion'],'v3.1_early_arithmetic_scaffold')
            key=(new['criterion'],new['semantic_id'])
            if key in existing:continue
            new['recovery_case']='scaffold/'+subcase(r['ir']);new['split_group_id']=new['semantic_id']
            existing.add(key);extra.append(new)
        rows.extend(extra);added[split]=len(extra)
    maximum=0;tok=MathTokenizer()
    for rows in groups.values():
        for r in rows:
            if not all(score(r['target'],r).values()):raise ValueError('Invalid gold target: '+r['semantic_id'])
            size=len(tok.prefix(r['prompt']))+len(tok.encode(r['target']))+1
            maximum=max(maximum,size)
            if size>2048:raise ValueError('Context overflow')
    shutil.copytree(source,dest)
    for split,rows in groups.items():(dest/(split+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
    for stage in manifest['stages']:
        idx=stage['index'];train=[r for r in groups['train'] if r['stage']==idx]
        stage['train_records']=len(train);stage['unique_train_tasks']=len({r['semantic_id'] for r in train})
        stage['validation_records']=sum(r['stage']==idx for r in groups['validation'])
        stage['criteria']=sorted({r['criterion'] for r in train})
        (dest/stage['remediation']).write_text(''.join(canonical(r)+'\n' for r in train),encoding='utf-8')
    manifest.update(revision='V3.1',parent_manifest_sha256=file_hash(source/'manifest.json'),records={s:len(rs) for s,rs in groups.items()})
    manifest['files']={name:file_hash(dest/name) for name in manifest['files']}
    (dest/'manifest.json').write_text(canonical(manifest),encoding='utf-8')
    audit={'revision':'V3.1','records':manifest['records'],'new_scaffolds':dict(added),'additional_remediation_records':merged,'max_tokens':maximum,'excluded_train_variants_matching_heldout':skipped,'source':str(source)}
    (dest/'v31_audit.json').write_text(canonical(audit));print(json.dumps(audit))
    return verify_manifest(dest)

if __name__=='__main__':build('G:/ctfn-text/data/v3_math_recovery_v5','G:/ctfn-text/data/v3_1_main')
