"""Automatic multi-domain learning experiment; never publishes accepted weights."""
import argparse
import itertools
import gc
import json
import time
from pathlib import Path
import torch
from .config import Config,canonical,identity
from .data import read_rows,example,composition,file_hash

DOMAINS=('math','retrieval','code','formal_logic')
BOOLQ_REV='35b264d03638db9f4ce671b711558bf7ff0f80d5'


def record(tower,prompt,target,source):
    return dict(id=identity([source,prompt]),semantic_id=identity(prompt),tower=tower,language='en',
        prompt=prompt,target=target,reference=target,source=source,criterion=source,
        verifier='experimental_teacher_not_truth_verified',verified=False,
        routing={'targets':[tower],'rounds':{tower:0}})


def prepare(root):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from .teacher_cycles import prepare as gsm_prepare
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        manifest=json.loads((root/'manifest.json').read_text())
        for name,digest in manifest['files'].items():
            if file_hash(root/name)!=digest:raise ValueError('experiment data changed')
        return
    gsm_prepare(root/'gsm8k')
    groups={t:[] for t in DOMAINS}
    for r in read_rows(root/'gsm8k/train.jsonl'):
        groups['math'].append(record('math',r['prompt'],r['target'],'gsm8k'))
    path=hf_hub_download('google/boolq','data/train-00000-of-00001.parquet',repo_type='dataset',revision=BOOLQ_REV,local_dir=root/'boolq')
    for r in pq.read_table(path).to_pylist():
        if len(r['passage'])>1800:continue
        groups['retrieval'].append(record('retrieval',r['passage']+'\nQuestion: '+r['question']+'\nAnswer only yes or no.',
            'yes' if r['answer'] else 'no','boolq'))
    for i in range(512):
        a=2+i%97;b=3+i//97
        groups['code'].append(record('code',f'Write only a Python function solve(x) that returns x * {a} + {b}.',
            f'def solve(x):\n    return x * {a} + {b}','synthetic_python'))
        groups['formal_logic'].append(record('formal_logic',f'Given P({i}), P(x) implies Q(x), Q(x) implies R(x). List the three propositions proving R({i}), separated by semicolons.',
            f'P({i});Q({i});R({i})','synthetic_logic'))
    files={}
    for tower,items in groups.items():
        items=list({r['semantic_id']:r for r in items}.values())
        items.sort(key=lambda r:identity(r['id']))
        for split,subset in [('heldout',items[:32]),('train',items[32:])]:
            out=root/f'{tower}_{split}.jsonl';out.write_text(''.join(canonical(r)+'\n' for r in subset),encoding='utf-8');files[out.name]=file_hash(out)
    (root/'manifest.json').write_text(canonical({'files':files,'sources':{
        'gsm8k':{'repo':'openai/gsm8k','revision':'740312add88f781978c0658806c59bc2815b9866','license':'MIT'},
        'boolq':{'repo':'google/boolq','revision':BOOLQ_REV,'license':'CC-BY-SA-3.0'},
        'synthetic':'local bounded Python, logic and Math-to-String tasks'},
        'scope':'Learning experiment only. Teacher answers can be wrong. Held-out questions never enter training.'}))


def authorized_record(r):
    # Explicit experimental caller only. Default verification/ingestion is unchanged.
    return r.get('tower') in DOMAINS and bool(r.get('prompt')) and bool(r.get('target')) and r.get('teacher_revision')==Config().revision


def teach(rows,path,writer):
    if path.exists():return read_rows(path)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    config=Config()
    tokenizer=AutoTokenizer.from_pretrained(config.coordinator,revision=config.revision)
    model=AutoModelForCausalLM.from_pretrained(config.coordinator,revision=config.revision,torch_dtype=torch.bfloat16).to('cuda').eval()
    output=[]
    try:
        for i,r in enumerate(rows):
            ids=tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda')
            with torch.no_grad():answer=model.generate(ids,max_new_tokens=128,do_sample=False)
            text=tokenizer.decode(answer[0,ids.shape[1]:],skip_special_tokens=True).strip()
            if text:output.append({**r,'target':text,'teacher_revision':config.revision,'verified':False})
            writer({'phase':'teacher','state':'generating','targets':[r['tower']],'completed':i+1,'total':len(rows)})
    finally:
        del model
        gc.collect();torch.cuda.empty_cache()
    tmp=path.with_suffix('.tmp');tmp.write_text(''.join(canonical(r)+'\n' for r in output),encoding='utf-8');tmp.replace(path)
    return output


def measure(model,rows,tower):
    from .training import supervised_loss
    values=[];samples=[]
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for r in rows:
            reference={**r,'target':r.get('reference',r['target'])}
            values.append(float(supervised_loss(model,reference,tower)))
        for r in rows[:8]:
            answer=model.generate(r['prompt']+'\n',tower,max_tokens=128)
            samples.append({'prompt':r['prompt'],'expected':r.get('reference',r['target']),'output':answer,
                'exact_reference_match':answer.strip()==r.get('reference',r['target']).strip()})
    return {'reference_loss':sum(values)/len(values),'exact_reference_matches':sum(r['exact_reference_match'] for r in samples),
        'sample_count':len(samples),'samples':samples,'note':'Exact reference match is conservative, especially for equivalent code.'}


def run(root,data):
    from .artifact import load_bundle,save_bundle
    from .training import train,make_plan
    from .runtime import status_writer
    from .live import GPULock
    from .evaluation import evaluate
    root=Path(root);root.mkdir(parents=True,exist_ok=True);data=Path(data)
    writer=status_writer(root.parent);checkpoint=root/'current.cftn'
    with GPULock(root.parent):
        # Metadata is committed atomically with candidate weights.
        import zipfile
        with zipfile.ZipFile(checkpoint if checkpoint.exists() else root.parent/'tower_repairs/current.cftn') as z:
            meta=json.loads(z.read('manifest.json'))['metadata']
        completed=meta.get('experiment_completed',[])
        from .experiment_controls import pending,answer_pending,teaching,consumed
        consumed(root,meta.get('feedback_consumed',[]))
        def service_pause():
            while (root/'PAUSED').exists():
                writer({'state':'paused','phase':'experiment','scope':'Learning paused; queued questions are still answered'})
                if pending(root):
                    test_model,_,_=load_bundle(checkpoint if checkpoint.exists() else root.parent/'tower_repairs/current.cftn','cuda')
                    answer_pending(root,test_model,'paused checkpoint')
                    del test_model;gc.collect();torch.cuda.empty_cache()
                time.sleep(10)
        start_cycle=max([int(k.split(':')[0]) for k in completed],default=0)
        for cycle in itertools.count(start_cycle):
            service_pause()
            for tower in DOMAINS:
                key=f'{cycle}:{tower}'
                if key in completed:continue
                service_pause()
                pool=read_rows(data/f'{tower}_train.jsonl')
                batch=cycle%((len(pool)+47)//48)
                raw=pool[batch*48:(batch+1)*48]
                taught=teach(raw,data/f'teacher_{batch}_{tower}.jsonl',writer)
                feedback=teaching(root,tower)
                # User test questions stay out of the fixed held-out panel.
                heldout_ids={r['semantic_id'] for r in read_rows(data/f'{tower}_heldout.jsonl')}
                feedback=[r for r in feedback if r['semantic_id'] not in heldout_ids]
                taught=taught+feedback
                if not taught:raise ValueError('teacher produced no nonempty answers')
                writer({'phase':'specialist','state':'starting','targets':[tower],'scope':'Experimental continual learning'})
                model,_,_=load_bundle(checkpoint if checkpoint.exists() else root.parent/'tower_repairs/current.cftn','cuda')
                answer_pending(root,model,f'round {cycle+1} before {tower}')
                heldout=read_rows(data/f'{tower}_heldout.jsonl')[:16]
                before=measure(model,heldout,tower)
                replay=[r for r in read_rows('data/train.jsonl') if r['tower']==tower and r['language']=='en' and not r.get('specialist_targets')]
                if cycle:
                    replay+=read_rows(data/f'teacher_0_{tower}.jsonl')
                plan=make_plan('continual',(tower,),taught,verifier=authorized_record)
                state=train(model,taught,plan,50,replay=replay,status=lambda m:writer({**m,'round':cycle+1,'stage_steps':50}),verifier=authorized_record)
                after=measure(model,heldout,tower)
                report={'before':before,'after':after,'teacher_reference_agreement':sum(r['target'].strip()==r['reference'].strip() for r in taught)/len(taught),
                    'training_examples':len(taught),'steps':50,'isolation':state['frozen_hashes_verified'],'accepted_release':False}
                completed.append(key)
                save_bundle(checkpoint,model,training=state,metadata={'experimental':True,'experiment_completed':completed,'last_report':report,'feedback_consumed':[r['feedback_id'] for r in feedback]})
                consumed(root,[r['feedback_id'] for r in feedback])
                answer_pending(root,model,f'round {cycle+1} after {tower}')
                (root/'latest.json').write_text(canonical({'round':cycle+1,'tower':tower,**report}))
                (root/f'{cycle}_{tower}.json').write_text(canonical(report))
                del model,state;gc.collect();torch.cuda.empty_cache()
            key=f'{cycle}:communication'
            if key not in completed:
                service_pause()
                model,_,_=load_bundle(checkpoint,'cuda')
                from .delegation import train_delegation
                report,state=train_delegation(model,cycle,writer)
                completed.append(key)
                save_bundle(checkpoint,model,training=state,metadata={'experimental':True,'experiment_completed':completed,'last_report':report})
                (root/f'{cycle}_communication.json').write_text(canonical(report))
                (root/'latest.json').write_text(canonical({'round':cycle+1,'tower':'communication',**report}))
                answer_pending(root,model,f'round {cycle+1} communication')
                del model,state;gc.collect();torch.cuda.empty_cache()
        writer({'state':'finished','phase':'experiment','result':{'completed':completed,'release_activated':False}})


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--wait',action='store_true');a=p.parse_args()
    prepare('data/learning_experiment')
    if a.prepare_only:return
    if a.wait:
        while not Path('artifacts/tower_repairs/final.json').exists() or Path('artifacts/gpu.lock').exists():time.sleep(30)
    if not Path('artifacts/tower_repairs/final.json').exists():raise RuntimeError('Wait for repairs to finish')
    try:
        run('artifacts/learning_experiment','data/learning_experiment')
    except Exception as exc:
        from .runtime import status_writer
        status_writer('artifacts')({'state':'failed','phase':'experiment','error':str(exc)})
        raise


if __name__=='__main__':main()
