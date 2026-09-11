import json
from pathlib import Path
root=Path('G:/ctfn-text/artifacts/v3_1/sigreg_probe_round_991')
old={r['prompt'] for r in map(json.loads,Path('G:/ctfn-text/data/v3_1_main/validation.jsonl').open()) if r['stage']==0 and r['criterion']=='1NPV-2' and r['answer']=='='}
summary={}
for arm in ('baseline','sigreg'):
    r=json.loads((root/arm/'result.json').read_text());v=r['evaluation'];samples=[s for s in v['samples'] if s['prompt'] in old]
    summary[arm]={'coefficient':r['coefficient'],'answers':v['accuracy'],'traces':v['trace_accuracy'],'correct':sum(s['answer_correct'] for s in v['samples']),'total':v['examples'],'original_equality_correct':sum(s['answer_correct'] for s in samples),'original_equality_total':len(samples),'comparison':v['criteria']['1NPV-2'],'round_losses':r['losses']}
(root/'summary.json').write_text(json.dumps(summary));print(json.dumps(summary))
