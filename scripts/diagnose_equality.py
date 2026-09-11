import json,torch
from pathlib import Path
from cftn_v3.local_specialist import load_specialist
from cftn_v3.criterion_curriculum_training import evaluate
root=Path('G:/ctfn-text/artifacts/v3_1/math')
assert json.loads((root/'status.json').read_text())['state']=='paused'
source=Path('G:/ctfn-text/data/v3_1_main');panels={}
for split in ('train','validation'):
    panels[split]=[r for r in map(json.loads,(source/(split+'.jsonl')).open()) if r['stage']==0 and r['criterion']=='1NPV-2' and r['answer']=='=']
panels['fresh']=json.loads(Path('G:/ctfn-text/data/v3_1_equality_v2/equality_diagnostic.json').read_text())
model,saved=load_specialist(root/'current.specialist');model.to('cuda');torch.set_num_threads(4)
results={'checkpoint_round':saved['metadata']['round'],'panels':{}}
for name,rows in panels.items():
    print('Evaluating',name,len(rows),flush=True)
    results['panels'][name]=evaluate(model,rows)
    print(name,results['panels'][name]['accuracy'],flush=True)
(root/'equality_diagnostic_before.json').write_text(json.dumps(results))
