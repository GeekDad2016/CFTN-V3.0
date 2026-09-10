"""Create an immutable recovery derivative, preserving every original split."""
import json,random,collections,shutil
from pathlib import Path
from .full_curriculum_data import make,verify_manifest
from .data import file_hash
from .config import canonical
from .local_specialist import MathTokenizer
from .math_procedures import score

def family(ir):
    op=ir['op']
    if op in ('add','multiply') and 'left' in ir:return (op,*sorted((ir['left'],ir['right'])))
    if op=='subtract':return (op,ir['left'],ir['right'])
    if op=='divide':return (op,ir['dividend'],ir['divisor'])
    return canonical({k:v for k,v in ir.items() if k not in ('type','detail','strategy')})

def subcase(ir):
    if ir['op']=='divide':return 'zero_quotient_digit' if '0' in str(ir['dividend']//ir['divisor']) else 'nonzero_quotient_digits'
    if ir['op']=='subtract':return 'zero_borrow' if '0' in str(ir['left']) else 'other_borrow'
    a,b=ir['left'],ir['right']
    if ir['op']=='multiply':a,b=a*(b%10),a*(b//10*10)
    carry=0;count=0
    for i in range(max(len(str(a)),len(str(b)))):
        carry=int(a//10**i%10+b//10**i%10+carry>=10);count+=carry
    return 'multiple_carries' if count>=2 else 'single_or_no_carry'

def build(source,dest):
    source,dest=Path(source),Path(dest)
    if dest.exists():return verify_manifest(dest)
    manifest=verify_manifest(source);rng=random.Random(20260910)
    groups={s:[json.loads(x) for x in (source/(s+'.jsonl')).open()] for s in ('train','validation','test')}
    occupied={family(r['ir']) for rows in groups.values() for r in rows}
    qpath=source/'quarantined_heldout.jsonl'
    if qpath.exists():occupied.update(family(r['ir']) for r in map(json.loads,qpath.open()) if 'ir' in r)
    additions={'train':[],'validation':[]};counts=collections.Counter()
    criteria=['RECOVERY-PARTIAL-SUM','EXT-MULTI-DIGIT-SUBTRACT','KS2-EXACT-DIVIDE','KS2-LONG-MULTIPLY']
    for criterion in criteria:
        for split,total in [('validation',128),('train',4000)]:
            n=0
            while n<total:
                if criterion=='RECOVERY-PARTIAL-SUM':
                    a,b=rng.randint(100,999),rng.randint(10,99)
                    ir={'op':'add','left':a*(b%10),'right':a*(b//10*10)}
                elif criterion=='EXT-MULTI-DIGIT-SUBTRACT':
                    a=rng.randint(100,9999)
                    if n%2==0:a=(a//100)*100
                    ir={'op':'subtract','left':a,'right':rng.randint(1,a-1)}
                elif criterion=='KS2-EXACT-DIVIDE':
                    b=rng.randint(2,99);q=rng.randint(10,499)
                    if n%2==0:q=(q//10)*10
                    ir={'op':'divide','dividend':b*q,'divisor':b}
                else:ir={'op':'multiply','left':rng.randint(100,999),'right':rng.randint(10,99)}
                key=family(ir)
                if key in occupied:continue
                occupied.add(key);n+=1
                # Both target forms stay in the same split; no scaffold leaks held-out tasks.
                for detail in (None,'place_value_v1'):
                    row=make({**ir,**({'detail':detail} if detail else {})},5,criterion)
                    row['recovery_case']=('scaffold' if detail else 'direct')+'/'+subcase(ir)
                    row['split_group_id']=row['semantic_id']
                    row['source_record']='arithmetic_recovery_v1'
                    additions[split].append(row);counts[criterion]+=split=='train'
    tok=MathTokenizer();maximum=0
    for rows in additions.values():
        for r in rows:
            if not all(score(r['target'],r).values()):raise ValueError('Gold contract failure')
            size=len(tok.prefix(r['prompt']))+len(tok.encode(r['target']))+1
            maximum=max(maximum,size)
            if size>2048:raise ValueError('Token overflow')
    shutil.copytree(source,dest)
    for split,rows in additions.items():
        groups[split]+=rows
        (dest/(split+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in groups[split]),encoding='utf-8')
    recovery=[r for r in groups['train'] if r['stage']==5]
    (dest/'remediation_05.jsonl').write_text(''.join(canonical(r)+'\n' for r in recovery),encoding='utf-8')
    manifest['parent_manifest_sha256']=file_hash(source/'manifest.json')
    manifest['recovery_release']='arithmetic_recovery_v1';manifest['records']={s:len(rs) for s,rs in groups.items()}
    stage=manifest['stages'][5];stage['criteria']=sorted(set(stage['criteria']+criteria));stage['train_records']=len(recovery)
    stage['validation_records']=sum(r['stage']==5 for r in groups['validation']);stage['unique_train_tasks']=len({r['semantic_id'] for r in recovery})
    stage['scope']='Multi-digit arithmetic with partial-product addition and place-value recovery scaffolds'
    manifest['files']={name:file_hash(dest/name) for name in manifest['files']}
    (dest/'manifest.json').write_text(canonical(manifest),encoding='utf-8')
    audit={'added':{s:len(rs) for s,rs in additions.items()},'max_tokens':maximum,'training_records_by_criterion':dict(counts),'heldout_family_overlap':0,'original_splits_preserved':True}
    (dest/'arithmetic_recovery_audit.json').write_text(canonical(audit));print(json.dumps(audit))
    return verify_manifest(dest)

if __name__=='__main__':
    build('G:/ctfn-text/data/v3_full_math_v4','G:/ctfn-text/data/v3_math_recovery_v5')
