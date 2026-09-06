"""Same-range addition curriculum and bounded, mastery-checked recovery."""
import argparse
import json
import os
import random
import time
from pathlib import Path
from .config import canonical, identity
from .data import read_rows, file_hash

SOURCE = 'cftn_addition_repair_v1'


def make_row(a, b):
    if not (0 <= a < 10000 and 0 <= b < 10000): raise ValueError('operand bounds')
    return dict(id=identity([SOURCE,a,b]), semantic_id=identity([SOURCE,min(a,b),max(a,b)]),
        source=SOURCE, tower='math', language='en', a=a, b=b,
        prompt=f'Calculate {a}+{b}.', target=f'<work>{a}+{b}={a+b}</work><answer>{a+b}</answer>',
        criterion='carry' if a%10+b%10>=10 else 'no_carry', verified=True,
        verifier='exact_addition_v1', routing={'targets':['math'],'rounds':{'math':0}})


def verify(row, output=None):
    try:
        a,b=row['a'],row['b']
        if type(a) is not int or type(b) is not int: return False
        expected=make_row(a,b)
        return all(row.get(k)==expected[k] for k in ('id','semantic_id','source','tower','language','prompt','criterion','verifier')) and (row['target'] if output is None else output).strip()==expected['target']
    except (ValueError,KeyError,TypeError): return False


def curriculum():
    # Split unordered operand pairs, so reversed copies cannot leak across panels.
    result={k:[] for k in ('train','development','test','extrapolation')}
    for a in range(100):
        for b in range(a,100):
            bucket=int(identity([SOURCE,a,b])[:8],16)%10
            split='test' if bucket==0 else 'development' if bucket==1 else 'train'
            result[split].append(make_row(a,b))
    for a in range(1000,1032): result['extrapolation'].append(make_row(a,195))
    return result


def panel(rows,count=64):
    groups=[[r for r in rows if r['criterion']==c] for c in ('carry','no_carry')]
    for group in groups: group.sort(key=lambda r:identity(r['id']))
    return [r for pair in zip(*groups) for r in pair][:count]


def prepare(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    rows=curriculum()
    manifest_path=path/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['generator']!=file_hash(Path(__file__)): raise ValueError('repair generator changed')
        for split in rows:
            if file_hash(path/f'{split}.jsonl')!=manifest['files'][split]['sha256']: raise ValueError('repair dataset changed')
        return rows
    files={}
    for split,items in rows.items():
        out=path/f'{split}.jsonl';out.write_text(''.join(canonical(r)+'\n' for r in items),encoding='utf-8')
        files[split]={'count':len(items),'sha256':file_hash(out)}
    manifest_path.write_text(canonical({'generator':file_hash(Path(__file__)),'files':files,
        'scope':'English addition 0-99, pair-disjoint same-range splits; extrapolation separate'}))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='artifacts/math_repair')
    p.add_argument('--bundle',default='artifacts/bootstrap.cftn')
    p.add_argument('--data',default='data/math_repair')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    rows=prepare(args.data)
    if args.prepare_only:
        print(canonical({k:len(v) for k,v in rows.items()}));return
    from .artifact import load_bundle,save_bundle
    from .training import train,make_plan
    from .evaluation import evaluate
    from .runtime import status_writer
    from .live import GPULock
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    writer=status_writer(root.parent)
    current=root/'current.cftn'
    report_path=root/'progress.json'
    with GPULock(root.parent):
        try:
            writer({'phase':'specialist','state':'starting','targets':['math'],'scope':'Same-range Math repair'})
            model,state,meta=load_bundle(current if current.exists() else args.bundle,'cuda',training=current.exists())
            old=json.loads(report_path.read_text()) if report_path.exists() else {}
            if current.exists() and meta.get('repair_source')!=SOURCE: raise ValueError('wrong resume checkpoint')
            # Replay excludes all development/test operand pairs, including reversed forms.
            heldout={(min(r['a'],r['b']),max(r['a'],r['b'])) for split in ('development','test') for r in rows[split]}
            import re
            replay=[]
            for r in read_rows('data/train.jsonl'):
                if r['tower']!='math' or r.get('specialist_targets') or r['language']!='en':continue
                match=re.fullmatch(r'Calculate (\d+)\+(\d+)\.',r['prompt'])
                if match and tuple(sorted(map(int,match.groups()))) not in heldout:replay.append(r)
            retention=sorted(replay,key=lambda r:identity(r['id']))[:32]
            retention_ids={r['id'] for r in retention}
            replay=[r for r in replay if r['id'] not in retention_ids]
            retention_prompts={r['prompt'] for r in retention}
            acquisition=[r for r in rows['train'] if r['prompt'] not in retention_prompts]
            def score(items):
                report=evaluate(model,items,max_tokens=64)
                return {'correct':sum(o['correct'] for o in report['outputs']),'count':len(report['outputs']),
                        'samples':[dict(prompt=r['prompt'],expected=r['target'],**o) for r,o in zip(items[:4],report['outputs'][:4])]}
            baseline=meta.get('baseline') if current.exists() else None
            if baseline is None: baseline=score(retention)
            passes=meta.get('passes',0) if current.exists() else 0
            plan=make_plan('specialist',('math',),acquisition)
            start=state['step'] if state else 0
            progress=meta.get('progress',old)
            if start>=1000 or passes>=2:
                writer({'phase':'specialist','state':'finished','targets':['math'],'result':progress,'next':'Review saved results; bounded run already finished'})
                return
            for step in range(start,1000,100):
                state=train(model,acquisition,plan,100,replay=replay,state=state,status=writer)
                writer({'phase':'evaluation','state':'evaluating','targets':['math'],'completed':0,'total':96})
                dev,ret=score(panel(rows['development'])),score(retention)
                passed=dev['correct']/dev['count']>=.9 and ret['correct']>=baseline['correct']
                passes=passes+1 if passed else 0
                progress={'step':state['step'],'development':dev,'retention':ret,'baseline_retention':baseline,
                    'consecutive_passes':passes,'scope':'Same-range Math recovery; not release activation'}
                # Resume metadata and weights commit together before publishing progress.
                writer({'phase':'specialist','state':'saving','targets':['math'],'step':state['step']})
                save_bundle(current,model,training=state,metadata={'repair_source':SOURCE,'baseline':baseline,'passes':passes,'progress':progress})
                report_path.write_text(canonical(progress))
                tower_path=root.parent/'tower_evaluations.json'
                reports=json.loads(tower_path.read_text()) if tower_path.exists() else {}
                reports['math']={'groups':{'math:en':{'correct':dev['correct'],'count':dev['count'],'accuracy':dev['correct']/dev['count']}},
                    'samples':dev['samples'],'scope':progress['scope'],'completed':dev['count'],'total':dev['count']}
                temp=tower_path.with_suffix('.tmp');temp.write_text(canonical(reports));temp.replace(tower_path)
                if passes>=2:
                    progress['final_test']=score(panel(rows['test']))
                    progress['extrapolation']=score(rows['extrapolation'])
                    progress['mastered']=progress['final_test']['correct']/progress['final_test']['count']>=.9
                    report_path.write_text(canonical(progress))
                    break
            writer({'phase':'specialist','state':'finished','targets':['math'],'result':progress,
                    'next':'Review mastery before any further tower or integration training'})
        except Exception as error:
            writer({'phase':'specialist','state':'failed','targets':['math'],'error':str(error)})
            raise


if __name__=='__main__': main()
