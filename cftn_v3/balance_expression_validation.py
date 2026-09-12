"""Add held-out expression comparisons without changing training or old validation."""
import json,collections,random,shutil
from pathlib import Path
from .full_curriculum_data import make,verify_manifest
from .balanced_curriculum_data import comparison_family
from .config import canonical
from .data import file_hash
from .math_procedures import score
from .local_specialist import MathTokenizer

def build(source,dest):
    source,dest=Path(source),Path(dest);assert not dest.exists()
    m=verify_manifest(source);groups={s:list(map(json.loads,(source/(s+'.jsonl')).open())) for s in ('train','validation','test')}
    used={comparison_family(r['ir']) for rs in groups.values() for r in rs}
    q=source/'quarantined_heldout.jsonl'
    if q.exists():used.update(comparison_family(r['ir']) for r in map(json.loads,q.open()) if 'ir' in r)
    current=[r for r in groups['validation'] if r['stage']==1 and r['criterion']=='EXT-COMPARE-EXPRESSIONS']
    counts=collections.Counter(r['answer'] for r in current);target=max(counts.values());rng=random.Random(312012);added=[];ops=('add','subtract')
    def operands(v,op):
        a=rng.randint(0,v) if op=='add' else rng.randint(0,20-v)
        return [a,v-a] if op=='add' else [v+a,a]
    for label in ('<','=','>'):
        n=0;tries=0
        while counts[label]<target:
            tries+=1;assert tries<1000000,'Insufficient held-out families'
            lo=ops[n%2];ro=ops[(n//2)%2];a=rng.randint(0,20)
            b=a if label=='=' else rng.randint(a+1,20) if a<20 else 20
            if label!='=' and a==b:continue
            if label=='>':a,b=b,a
            ir={'op':'compare_expressions','left':operands(a,lo),'right':operands(b,ro),'left_op':lo,'right_op':ro}
            normalized=lambda op,vs:(op,tuple(sorted(vs) if op=='add' else vs))
            if normalized(lo,ir['left'])==normalized(ro,ir['right']):continue
            family=comparison_family(ir)
            if family in used:continue
            used.add(family);r=make(ir,1,'EXT-COMPARE-EXPRESSIONS','balanced_expression_validation_v1')
            assert r['answer']==label and all(score(r['target'],r).values())
            assert len(MathTokenizer().prefix(r['prompt']))+len(MathTokenizer().encode(r['target']))+1<=2048
            added.append(r);counts[label]+=1;n+=1
    shutil.copytree(source,dest);groups['validation']+=added
    (dest/'validation.jsonl').write_text(''.join(canonical(r)+'\n' for r in groups['validation']),encoding='utf-8')
    m['parent_manifest_sha256']=file_hash(source/'manifest.json');m['records']['validation']=len(groups['validation'])
    m['stages'][1]['validation_records']=sum(r['stage']==1 for r in groups['validation'])
    m['files']={name:file_hash(dest/name) for name in m['files']};(dest/'manifest.json').write_text(canonical(m))
    audit={'added':dict(collections.Counter(r['answer'] for r in added)),'final_counts':dict(counts),'new_families':len(added),'training_unchanged':file_hash(source/'train.jsonl')==file_hash(dest/'train.jsonl'),'test_unchanged':file_hash(source/'test.jsonl')==file_hash(dest/'test.jsonl')}
    (dest/'expression_validation_audit.json').write_text(canonical(audit));print(json.dumps(audit))
if __name__=='__main__':build('G:/ctfn-text/data/v3_1_foundation_v3','G:/ctfn-text/data/v3_1_balanced_validation_v4')
