"""Sequential bounded repairs; mastery failures never become release approval."""
import json
import os
import random
from pathlib import Path
from .config import TOWERS, canonical, identity
from .data import example, read_rows

SOURCE='cftn_tower_repair_v1'
ORDER=('string',)+tuple(t for t in TOWERS if t not in ('math','string'))


def row(tower,index):
    r=example(tower,index,'en')
    if tower=='string':
        rng=random.Random(index)
        text=''.join(rng.choice('abcXYZ019 -șțăî') for _ in range(4+index%21))
        r.update(prompt='Reverse exactly: '+text,target=text[::-1],criterion='varied_unicode_reverse')
    if tower=='commonsense':
        places=('table','cupboard','desk','shelf')
        place=places[index%4]
        r.update(prompt=f'Case {index}: Ana leaves a book on the {place}. Nobody moves it. Where is the book? Answer with the location only.',
                 target=place,criterion='varied_object_location')
    r.update(source=SOURCE,verifier='tower_repair_exact_v1')
    r['id']=identity([SOURCE,tower,index]);r['semantic_id']=identity([tower,r['prompt']])
    return r


def verify(r,output=None):
    try:
        expected=row(r['tower'],r['index'])
        if any(r.get(k)!=expected[k] for k in ('id','source','prompt','semantic_id','verifier','language')):return False
        if r['tower'] in ('string','commonsense'):
            return (r['target'] if output is None else output)==expected['target']
        from .data import verify as original_verify
        return original_verify(example(r['tower'],r['index'],'en'),r['target'] if output is None else output)
    except (KeyError,ValueError,TypeError):return False


def dataset(tower):
    result={s:[] for s in ('train','development','test')}
    seen=set()
    for index in range(4096,8192):
        r=row(tower,index)
        if r['semantic_id'] in seen:continue
        seen.add(r['semantic_id'])
        bucket=int(identity(r['semantic_id'])[:8],16)%10
        result['test' if bucket==0 else 'development' if bucket==1 else 'train'].append(r)
    return result


def main():
    from .artifact import load_bundle,save_bundle
    from .runtime import status_writer
    from .live import GPULock
    from .training import train,make_plan
    from .evaluation import evaluate
    root=Path('artifacts/tower_repairs');root.mkdir(parents=True,exist_ok=True)
    current=root/'current.cftn';writer=status_writer(root.parent)
    with GPULock(root.parent):
        try:
            writer({'state':'starting','phase':'specialist','targets':['string'],'scope':'Automatic sequential tower repairs'})
            model,state,meta=load_bundle(current if current.exists() else 'artifacts/math_repair/current.cftn','cuda',training=current.exists())
            if current.exists() and meta.get('repair_version')!=SOURCE:raise ValueError('wrong repair checkpoint')
            completed=meta.get('completed',{})
            math_before=meta.get('math_before')
            def score(items):
                report=evaluate(model,items,max_tokens=128)
                report['samples']=[dict(prompt=r['prompt'],expected=r['target'],**o) for r,o in zip(items[:4],report['outputs'][:4])]
                report['accuracy']=sum(o['correct'] for o in report['outputs'])/len(report['outputs'])
                return report
            if math_before is None:
                from .math_repair import curriculum,panel
                math_before=score(panel(curriculum()['test'])[:16])
            for tower in ORDER:
                if tower in completed:continue
                data=dataset(tower)
                dest=Path('data/tower_repairs')/tower;dest.mkdir(parents=True,exist_ok=True)
                for split,items in data.items():
                    text=''.join(canonical(r)+'\n' for r in items)
                    path=dest/f'{split}.jsonl'
                    if path.exists() and path.read_text(encoding='utf-8')!=text:raise ValueError('repair dataset changed')
                    path.write_text(text,encoding='utf-8')
                resuming=meta.get('tower')==tower
                if not resuming:state=None
                passes=meta.get('passes',0) if resuming else 0
                start=state['step'] if state else 0
                replay=[r for r in read_rows('data/train.jsonl') if r['tower']==tower and r['language']=='en' and not r.get('specialist_targets')]
                plan=make_plan('specialist',(tower,),data['train'])
                threshold=.99 if tower=='string' else .9
                result=meta.get('last_result',{}) if resuming else {}
                for step in range(start,1000,100):
                    state=train(model,data['train'],plan,100,state=state,replay=replay,status=writer)
                    writer({'phase':'evaluation','state':'evaluating','targets':[tower],'completed':0,'total':64})
                    result=score(data['development'][:64])
                    passes=passes+1 if result['accuracy']>=threshold else 0
                    meta={'repair_version':SOURCE,'tower':tower,'passes':passes,'completed':completed,'math_before':math_before,'last_result':result}
                    writer({'phase':'specialist','state':'saving','targets':[tower],'step':state['step']})
                    save_bundle(current,model,training=state,metadata=meta)
                    tower_path=root.parent/'tower_evaluations.json'
                    reports=json.loads(tower_path.read_text()) if tower_path.exists() else {}
                    reports[tower]={**result,'scope':'Same-range repair development panel; not release approval'}
                    tmp=tower_path.with_suffix('.tmp');tmp.write_text(canonical(reports));tmp.replace(tower_path)
                    (root/'progress.json').write_text(canonical({'current':tower,'step':state['step'],'accuracy':result['accuracy'],'completed':completed}))
                    if passes>=2:break
                test=score(data['test'][:64])
                completed[tower]={'step':state['step'],'validation_accuracy':result['accuracy'],'test_accuracy':test['accuracy'],
                    'mastered':passes>=2 and test['accuracy']>=threshold,'scope':'Bounded template exercises only'}
                meta.update(completed=completed)
                save_bundle(current,model,training=state,metadata=meta)
                (root/'progress.json').write_text(canonical({'completed':completed,'last_tower':tower}))
                print(canonical({tower:completed[tower]}),flush=True)
            from .math_repair import curriculum,panel
            math_after=score(panel(curriculum()['test'])[:16])
            result={'completed':completed,'math_before':math_before['accuracy'],'math_after':math_after['accuracy'],
                'integration_started':False,'release_activated':False}
            (root/'final.json').write_text(canonical(result))
            writer({'phase':'specialist','state':'finished','result':result})
        except Exception as exc:
            writer({'phase':'specialist','state':'failed','error':str(exc)})
            raise


if __name__=='__main__':main()
