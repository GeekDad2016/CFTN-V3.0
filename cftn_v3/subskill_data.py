"""Training-only augmentation; no validation or test questions are added."""
import collections
import json
import random
import shutil
from pathlib import Path
from .balanced_curriculum_data import comparison_family
from .config import canonical
from .data import file_hash
from .full_curriculum_data import make, verify_manifest
from .local_specialist import MathTokenizer
from .math_procedures import score
from .subskill_recovery import bucket

def build(source, dest):
    source, dest = Path(source), Path(dest)
    assert not dest.exists()
    manifest = verify_manifest(source)
    read = lambda name: [json.loads(l) for l in (source/name).read_text().splitlines()]
    train = read('train.jsonl'); held = read('validation.jsonl')+read('test.jsonl')
    if (source/'quarantined_heldout.jsonl').exists():
        held += [r.get('record',r) for r in read('quarantined_heldout.jsonl')]
    reserved = {comparison_family(r['ir']) for r in held}
    seen = {r['semantic_id'] for r in train}
    rng = random.Random(312013); counts = collections.Counter(); added = []
    for _ in range(120000):
        ir = {'op':'compare_expressions'}
        for side in ('left','right'):
            op = rng.choice(('add','subtract')); v = rng.randrange(21)
            x = rng.randrange(v+1) if op == 'add' else rng.randrange(21-v)
            ir[side+'_op'] = op; ir[side] = [x,v-x] if op == 'add' else [v+x,x]
        row = make(ir,1,'EXT-COMPARE-EXPRESSIONS','subskill_recovery_v1')
        key = (row['answer'],bucket(row),ir['left_op'],ir['right_op'])
        if counts[key] >= 80 or row['semantic_id'] in seen or comparison_family(ir) in reserved: continue
        assert all(score(row['target'],row).values())
        assert len(MathTokenizer().prefix(row['prompt']))+len(MathTokenizer().encode(row['target']))+1 <= 2048
        seen.add(row['semantic_id']); counts[key]+=1; added.append(row)
    assert added
    # The finite borrowing domain already exists in the source: use focused sampling,
    # rather than duplicate records or importing held-out operand pairs.
    shutil.copytree(source,dest)
    with (dest/'train.jsonl').open('a',encoding='utf-8') as stream:
        for r in added: stream.write(canonical(r)+'\n')
    manifest['parent_manifest_sha256'] = file_hash(source/'manifest.json')
    manifest['records']['train'] += len(added)
    manifest['stages'][1]['train_records'] += len(added)
    manifest['files'] = {name:file_hash(dest/name) for name in manifest['files']}
    (dest/'manifest.json').write_text(canonical(manifest))
    audit = {'added':len(added),'buckets':{str(k):v for k,v in counts.items()},
             'validation_unchanged':file_hash(source/'validation.jsonl')==file_hash(dest/'validation.jsonl'),
             'test_unchanged':file_hash(source/'test.jsonl')==file_hash(dest/'test.jsonl')}
    (dest/'subskill_audit.json').write_text(canonical(audit))
    print(json.dumps({k:v for k,v in audit.items() if k!='buckets'}))

if __name__ == '__main__':
    build('G:/ctfn-text/data/v3_1_balanced_validation_v4','G:/ctfn-text/data/v3_1_subskill_v5')
