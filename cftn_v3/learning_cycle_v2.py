"""Resumable canonical repair and teacher learning with bounded evaluations."""
import gc
import itertools
import json
import time
import zipfile
from pathlib import Path
import torch

from .answer_contracts import CONTRACT_VERSION, canonical_answer, correct
from .config import Config, canonical
from .data import read_rows, file_hash
from .learning_experiment import DOMAINS, BLOCK_STEPS, CHECKPOINT_STEPS, FRESH_EXAMPLES, record, measure, teach


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(canonical(value),encoding='utf-8');temporary.replace(path)


def prepare_v2(old, new):
    old,new=Path(old),Path(new);new.mkdir(parents=True,exist_ok=True)
    if (new/'manifest.json').exists():
        manifest=json.loads((new/'manifest.json').read_text())
        for name,digest in manifest['files'].items():
            if file_hash(new/name)!=digest:raise ValueError('v2 dataset checksum mismatch')
        return
    files={}
    for tower in DOMAINS:
        for split in ('train','heldout'):
            rows=read_rows(old/f'{tower}_{split}.jsonl')
            # Preserve the existing evaluation split, including its semantic IDs.
            if split=='train' and tower in ('code','formal_logic'):
                for i in range(512,4096):
                    if tower=='code':
                        a,b=2+i%97,3+i//97
                        rows.append(record(tower,f'Write only a Python function solve(x) that returns x * {a} + {b}.',
                            f'def solve(x):\n    return x * {a} + {b}','synthetic_python'))
                    else:
                        rows.append(record(tower,f'Given P({i}), P(x) implies Q(x), Q(x) implies R(x). List the three propositions proving R({i}), separated by semicolons.',
                            f'P({i});Q({i});R({i})','synthetic_logic'))
            for r in rows:
                r['target']=r['reference']=canonical_answer(tower,r['reference'])
                r['contract_version']=CONTRACT_VERSION
            out=new/f'{tower}_{split}.jsonl'
            out.write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8');files[out.name]=file_hash(out)
    write_json(new/'manifest.json',{'version':CONTRACT_VERSION,'files':files,
        'sources':json.loads((old/'manifest.json').read_text())['sources'],
        'split_policy':'Original held-out IDs preserved; only synthetic training pool expanded.'})


def authorized(row):
    return (row.get('tower') in DOMAINS and row.get('contract_version')==CONTRACT_VERSION
        and row.get('teacher_revision')==Config().revision
        and bool(row.get('prompt')) and correct(row['tower'],row.get('target',''),row.get('reference','')))


def select_pool(pool, cycle):
    count=min(FRESH_EXAMPLES,len(pool))
    offset=cycle*FRESH_EXAMPLES%len(pool)
    return [pool[(offset+i)%len(pool)] for i in range(count)]


def run(root,data):
    from .artifact import load_bundle,save_bundle
    from .training import train,make_plan
    from .runtime import status_writer
    from .live import GPULock
    from .experiment_controls import answer_pending,teaching,consumed
    root,data=Path(root),Path(data)
    writer=status_writer(root.parent);checkpoint=root/'current.cftn'
    with GPULock(root.parent):
        source=checkpoint if checkpoint.exists() else root.parent/'tower_repairs/current.cftn'
        with zipfile.ZipFile(source) as z:meta=json.loads(z.read('manifest.json'))['metadata']
        active=meta.get('v2_active')
        writer({'phase':'loading','state':'starting','scope':'Canonical answer repair / 1000-step learning blocks'})
        model,state,_=load_bundle(source,'cuda',training=bool(active))
        if model.config.specialist_specs:
            raise RuntimeError('Assembled native specialists require typed-request coordination training; legacy word-problem loop is disabled')
        model.config.max_continual_steps=BLOCK_STEPS
        completed=meta.get('experiment_completed',[])
        last=max([int(k.split(':')[0]) for k in completed],default=-1)
        start=active['cycle'] if active else (last if 'v2_repair_done' in meta else last+1)
        repair_done=meta.get('v2_repair_done',[])
        metrics=meta.get('v2_metrics',{})
        consumed(root,meta.get('feedback_consumed',[]))

        def pause():
            while (root/'PAUSED').exists():
                writer({'phase':'experiment','state':'paused','scope':'Checkpoint saved; dashboard questions available'})
                answer_pending(root,model,'paused v2 checkpoint');time.sleep(10)

        def save(progress, report):
            writer({'phase':'checkpoint','state':'saving','targets':[progress['tower']],
                'round':progress['cycle']+1,'step':progress['step'],'stage_steps':BLOCK_STEPS})
            save_bundle(checkpoint,model,training=state,metadata={'experimental':True,
                'experiment_completed':completed,'v2_active':progress if progress['step']<BLOCK_STEPS else None,
                'v2_repair_done':repair_done,'v2_metrics':metrics,'last_report':report,
                'feedback_consumed':[r['feedback_id'] for r in taught if r.get('feedback_id')]})

        for cycle in itertools.count(start):
            for tower in DOMAINS:
                key=f'{cycle}:{tower}'
                if key in completed:continue
                pause()
                repairing=tower not in repair_done
                selected=data/f'v2_{cycle}_{tower}.jsonl'
                if not selected.exists():
                    pool=read_rows(data/f'{tower}_train.jsonl')
                    raw=select_pool(pool,cycle)
                    if repairing:
                        taught=[{**r,'teacher_revision':Config().revision,'supervision':'reference_format_repair',
                                 'teacher_accepted':False} for r in raw]
                    else:
                        taught=teach(raw,data/f'qwen_v2_{cycle}_{tower}.jsonl',writer)
                    heldout_ids={r['semantic_id'] for r in read_rows(data/f'{tower}_heldout.jsonl')}
                    for r in teaching(root,tower):
                        if r['semantic_id'] in heldout_ids:continue
                        try:
                            target=canonical_answer(tower,r['target'])
                            taught.append({**r,'target':target,'reference':target,'contract_version':CONTRACT_VERSION})
                        except (ValueError,SyntaxError):continue
                    selected.write_text(''.join(canonical(r)+'\n' for r in taught),encoding='utf-8')
                taught=read_rows(selected)
                if not taught or not all(authorized(r) for r in taught):raise ValueError('invalid canonical training pool')
                heldout=read_rows(data/f'{tower}_heldout.jsonl')[:16]
                replay=[r for r in read_rows('data/train.jsonl') if r['tower']==tower and r['language']=='en' and not r.get('specialist_targets')][:256]
                if tower=='math':
                    import re
                    replay=[{**r,'target':canonical_answer(tower,re.search(r'<answer>.*?</answer>',r['target'])[0])} for r in replay]
                # Prior validated pools supply retention of recently acquired skills too.
                previous=sorted(data.glob(f'v2_*_{tower}.jsonl'),key=lambda p:int(p.name.split('_')[1]))
                for p in previous[:-1][-2:]:replay+=read_rows(p)[:128]
                retention=replay[:8]
                if active and active['cycle']==cycle and active['tower']==tower:
                    if active['pool_hash']!=file_hash(selected):raise ValueError('resumed training pool changed')
                    if state is None or state['step']!=active['step']:raise ValueError('checkpoint optimizer progress mismatch')
                    progress=active
                else:
                    state=None
                    writer({'phase':'evaluation','state':'running','targets':[tower],'scope':'Before repair: held-out, trained and retention answers'})
                    progress={'cycle':cycle,'tower':tower,'step':0,'pool_hash':file_hash(selected),
                        'before':measure(model,heldout,tower),'seen_before':measure(model,taught[:4],tower),
                        'retention_before':measure(model,retention,tower),
                        'learning_phase':'format_repair' if repairing else 'teacher_continual'}
                plan=make_plan('continual',(tower,),taught,verifier=authorized)
                while progress['step']<BLOCK_STEPS:
                    state=train(model,taught,plan,min(CHECKPOINT_STEPS,BLOCK_STEPS-progress['step']),
                        replay=replay,state=state,verifier=authorized,
                        status=lambda m:writer({**m,'round':cycle+1,'stage_steps':BLOCK_STEPS,
                            'learning_phase':progress['learning_phase'],'training_examples':len(taught)}))
                    progress['step']=state['step']
                    writer({'phase':'evaluation','state':'running','targets':[tower],
                        'step':state['step'],'stage_steps':BLOCK_STEPS,'scope':'16 held-out, 4 trained and 8 retention examples'})
                    after=measure(model,heldout,tower)
                    seen=measure(model,taught[:4],tower)
                    retained=measure(model,retention,tower)
                    report={**progress,'after':after,'seen_after':seen,'retention_after':retained,
                        'training_examples':len(taught),'steps':state['step'],'stage_steps':BLOCK_STEPS,
                        'teacher_accepted':sum(r.get('teacher_accepted',False) for r in taught),
                        'reference_targets':sum(r.get('supervision')!='validated_teacher' for r in taught),
                        'teacher_reference_agreement':sum(correct(tower,r['target'],r['reference']) for r in taught)/len(taught),
                        'isolation':state['frozen_hashes_verified'],'accepted_release':False,
                        'trained_examples':taught[:4],'contract_version':CONTRACT_VERSION}
                    metrics[tower]={'accuracy':after['accuracy'],'retention_ok':retained['accuracy']>=progress['retention_before']['accuracy']}
                    if state['step']==BLOCK_STEPS:
                        completed.append(key)
                        if repairing and tower not in repair_done:repair_done.append(tower)
                    save(progress,report)
                    write_json(root/f'{cycle}_{tower}.json',report)
                    write_json(root/'latest.json',{'round':cycle+1,'tower':tower,**report})
                    consumed(root,[r['feedback_id'] for r in taught if r.get('feedback_id')])
                    answer_pending(root,model,f'round {cycle+1} {tower} step {state["step"]}')
                    pause()
                state=None;active=None;gc.collect();torch.cuda.empty_cache()
            # Preserve the coordinator and gates until all four native panels pass.
            ready=all(metrics.get(t,{}).get('accuracy',0)>=.75 and metrics[t]['retention_ok'] for t in DOMAINS)
            if ready and f'{cycle}:communication' not in completed:
                from .delegation import train_delegation
                report,state=train_delegation(model,cycle,writer)
                completed.append(f'{cycle}:communication')
                save_bundle(checkpoint,model,training=state,metadata={'experimental':True,'experiment_completed':completed,
                    'v2_repair_done':repair_done,'v2_metrics':metrics,'last_report':report})
                write_json(root/f'{cycle}_communication.json',report)
                state=None
            else:
                write_json(root/'delegation_gate.json',{'state':'waiting_for_native_accuracy',
                    'minimum_accuracy':.75,'metrics':metrics,'note':'Coordinator, Dispatcher and bridges preserved while specialists recover.'})
