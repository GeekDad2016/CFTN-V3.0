"""Local-only native training with gated stages and bounded automatic remediation."""
import argparse
import collections
import json
import os
import random
import time
from pathlib import Path
import torch
from .config import canonical
from .data import file_hash
from .full_curriculum_data import verify_manifest
from .local_math_training import atomic,batch_loss,read
from .local_specialist import MathTokenizer,load_specialist,save_specialist
from .math_procedures import score

def panel(rows,per_criterion):
    groups=collections.defaultdict(dict)
    for r in rows:groups[r['criterion']][r.get('split_group_id',r['semantic_id'])]=r
    return [r for key in sorted(groups) for r in sorted(groups[key].values(),key=lambda r:r['semantic_id'])[:per_criterion]]

def sample(rows,count,seed):
    groups=collections.defaultdict(dict);rng=random.Random(seed)
    for r in rows:groups[r['criterion']][r['semantic_id']]=r
    groups={k:list(v.values()) for k,v in groups.items()};keys=sorted(groups)
    return [rng.choice(groups[keys[i%len(keys)]]) for i in range(count)] if keys else []

def evaluate(model,rows,progress=None):
    samples=[];model.eval()
    for i,r in enumerate(rows):
        budget=min(model.context-len(MathTokenizer().prefix(r['prompt'])),len(r['target'].encode())+64)
        output,eos=model.generate(r['prompt'],budget)
        result=score(output,r,eos)
        samples.append({'prompt':r['prompt'],'expected':r['answer'],'expected_trace':r['target'],
            'output':output,'criterion':r['criterion'],'phase':r['phase'],'complete':eos,**result})
        if progress and (i%8==0 or i+1==len(rows)):progress(i+1,len(rows))
    def metrics(values):
        return {'examples':len(values),**{name:sum(r[key] for r in values)/len(values) if values else 1.
            for name,key in [('accuracy','answer_correct'),('trace_accuracy','trace_correct'),('format_accuracy','format_correct')]}}
    return {**metrics(samples),'criteria':{c:metrics([r for r in samples if r['criterion']==c]) for c in sorted({r['criterion'] for r in samples})},'samples':samples}

def failed_criteria(report,retention=False,baseline=None):
    return [c for c,m in report['criteria'].items() if m['accuracy']<max(.95,(baseline or {}).get(c,{}).get('accuracy',0))
        or m['format_accuracy']<.95 or (not retention and m['trace_accuracy']<.90)]

def batches(rows,budget=4096):
    # Bounded attention padding even when later procedures approach 2048 bytes.
    ordered=[]
    for i in range(0,len(rows),256):ordered.extend(sorted(rows[i:i+256],key=lambda r:len(r['prompt'])+len(r['target'])))
    batch=[];longest=0
    for r in ordered:
        n=len(MathTokenizer().prefix(r['prompt']))+len(MathTokenizer().encode(r['target']))+1
        if batch and (max(n,longest)*(len(batch)+1)>budget or len(batch)>=32):yield batch;batch=[];longest=0
        batch.append(r);longest=max(longest,n)
    if batch:yield batch

def training_rows(active,prior,repair,weak,retention_weak,count,seed,attempt):
    active_n=count*3//4 if prior else count
    if attempt and weak:
        targeted=[r for r in repair if r['criterion'] in weak] or repair
        rows=sample(targeted,active_n//2,seed)+sample(active,active_n-active_n//2,seed+1)
    else:rows=sample(active,active_n,seed)
    replay_n=count-active_n;targeted_prior=[r for r in prior if r['criterion'] in retention_weak]
    focused=replay_n//2 if attempt and targeted_prior else 0
    rows+=sample(targeted_prior,focused,seed+2)+sample(prior,replay_n-focused,seed+3)
    random.Random(seed).shuffle(rows);return rows

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);data=Path(args.data)
    # One lock shared by every local native specialist in this pipeline.
    lock=out.parent/'native_training.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,canonical({'pid':os.getpid(),'output':str(out)}).encode());os.close(fd)
    current={};started=time.time()
    def status(**kw):
        current.update(kw);atomic(out/'status.json',{'pid':os.getpid(),'updated':time.time(),'elapsed_seconds':time.time()-started,**current})
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for the local training run')
        manifest=verify_manifest(data);digest=file_hash(data/'manifest.json');atomic(out/'curriculum.json',manifest)
        train=read(data/'train.jsonl');dev=read(data/'validation.jsonl')
        latest=out/'current.specialist';model,saved=load_specialist(latest if latest.exists() else args.initial_checkpoint)
        tower=saved['tower'];resumed=latest.exists();meta=saved['metadata'] if resumed else {}
        policy={k:getattr(args,k) for k in ('normal_rounds','remediation_rounds','attempts','examples','lr')}
        policy.update(validation_examples=getattr(args,'validation_examples',12),retention_examples=getattr(args,'retention_examples',4))
        saved_policy={'validation_examples':12,'retention_examples':4,**meta.get('policy',{})}
        if resumed and (meta.get('dataset_hash')!=digest or saved_policy!=policy):raise ValueError('Resume data or policy changed')
        if meta.get('accepted'):status(state='complete',accepted=True);return
        if meta.get('terminal'):status(state='blocked',reason=meta['terminal']);return
        model.to('cuda');torch.manual_seed(9307);torch.set_num_threads(4)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
        if resumed and saved.get('optimizer'):
            optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        completed=meta.get('completed',[]);start_stage=meta.get('stage_index',0);start_round=meta.get('round',1)
        cursor=meta.get('cursor',0);consecutive=meta.get('consecutive',0);weak=meta.get('weak',[]);retention_weak=meta.get('retention_weak',[])
        maxround=args.normal_rounds+args.attempts*args.remediation_rounds
        status(state='starting',phase='curriculum validation',source=str(args.initial_checkpoint),dataset=str(data),
            parameters=sum(p.numel() for p in model.parameters()),gpu=torch.cuda.get_device_name(),
            checkpoint=str(latest),completed=completed,epochs=maxround,policy=policy,
            queue='String is waiting for Maths acceptance and GPU release')
        def save(index,round_,cursor_=0,**extra):
            nonlocal meta
            meta={'tower':tower,'accepted':False,'dataset_hash':digest,'policy':policy,'stage_index':index,'round':round_,
                'cursor':cursor_,'completed':completed,'consecutive':consecutive,'weak':weak,'retention_weak':retention_weak,
                'source_checkpoint':str(args.initial_checkpoint),**extra}
            save_specialist(latest,model,meta,optimizer)
        def ev(rows,label):
            status(state='evaluating',evaluation=label,evaluation_done=0,evaluation_total=len(rows))
            return evaluate(model,rows,lambda done,total:status(evaluation_done=done,evaluation_total=total))
        for index in range(start_stage,len(manifest['stages'])):
            stage=manifest['stages'][index];phase=stage['name'];active=[r for r in train if r['stage']==index]
            prior=[r for r in train if r['stage']<index];validation=[r for r in dev if r['stage']==index]
            active_panel=panel(validation,policy['validation_examples']);retention_panel=panel([r for r in dev if r['stage']<index],policy['retention_examples'])
            repair=read(data/stage['remediation']);entry_path=out/f'{phase}_before.json'
            status(phase=phase,stage_index=index,scope=stage['scope'],epoch=start_round if index==start_stage else 1,
                stage_count=len(manifest['stages']),active_examples=len(active),completed=completed)
            if entry_path.exists():entry=json.loads(entry_path.read_text())
            else:
                entry={'active':ev(active_panel,'stage baseline'),'retention':ev(retention_panel,'retention baseline')}
                atomic(entry_path,entry)
            for round_ in range(start_round if index==start_stage else 1,maxround+1):
                begun=time.time();attempt=max(0,(round_-args.normal_rounds-1)//args.remediation_rounds+1)
                status(epoch=round_,remediation_attempt=attempt,remediation_criteria=weak,step=cursor)
                seed=9307+index*100000+round_
                # Active training is 75%, prior accepted skills 25%; stage zero has no replay.
                rows=training_rows(active,prior,repair,weak,retention_weak,args.examples,seed,attempt)
                chunks=list(batches(rows));losses=[];model.train()
                status(state='remediating' if attempt else 'training',steps=len(chunks),evaluation=None)
                for step,chunk in enumerate(chunks):
                    if step<cursor:continue
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=batch_loss(model,chunk)
                    if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
                    if step%10==0:status(step=step+1,loss=losses[-1],gpu_bytes=torch.cuda.memory_allocated())
                    if (step+1)%100==0 or (out/'STOP').exists():save(index,round_,step+1)
                    if (out/'STOP').exists():status(state='paused',reason='Safe stop requested; optimizer and cursor saved');return
                cursor=0
                observed=ev(active_panel,'active validation');retained=ev(retention_panel,'prior-stage retention')
                weak=failed_criteria(observed);retention_weak=failed_criteria(retained,True,entry['retention']['criteria'])
                passed=not weak and not retention_weak;consecutive=consecutive+1 if passed else 0
                report={'phase':phase,'epoch':round_,'loss':sum(losses)/len(losses) if losses else current.get('loss'),
                    'active':observed,'retention':retained,'retention_baseline':entry['retention']['accuracy'],
                    'passed':passed,'consecutive':consecutive,'remediation_attempt':attempt,'failed_criteria':weak,
                    'retention_failed_criteria':retention_weak,'elapsed':time.time()-begun}
                atomic(out/f'{phase}_epoch_{round_:03d}.json',report)
                save(index,round_+1);status(state='evaluated',passed=passed,accuracy=observed['accuracy'],retention=retained['accuracy'])
                if consecutive>=2:
                    full=ev(panel(validation,100000),'complete stage validation')
                    cumulative=ev(panel([r for r in dev if r['stage']<index],12),'complete retention gate')
                    full_fail=failed_criteria(full);cum_fail=failed_criteria(cumulative,True)
                    atomic(out/f'{phase}_promotion_validation.json',{'active':full,'retention':cumulative,'passed':not full_fail and not cum_fail})
                    if not full_fail and not cum_fail:
                        completed.append(phase);consecutive=0;weak=[];retention_weak=[];save(index+1,1)
                        status(state='stage_complete',reason='Mastery and retention passed',completed=completed);break
                    weak=full_fail;retention_weak=cum_fail;consecutive=0;save(index,round_+1)
                if round_==args.normal_rounds or (round_>args.normal_rounds and (round_-args.normal_rounds)%args.remediation_rounds==0):
                    status(reason='Validation failed; targeted remediation queued' if round_<maxround else 'Remediation budget exhausted')
            else:
                save(index,maxround+1,terminal='Stage failed after all automatic remediation attempts')
                status(state='blocked',reason='Stage failed after all automatic remediation attempts; no stage skipped');return
            start_round=1;cursor=0
        # Tests never influence remediation. Once consumed, a failure requires a new release evaluation.
        if (out/'sealed_test_started.json').exists():
            status(state='blocked',reason='Sealed test already opened; manual review required');return
        atomic(out/'sealed_test_started.json',{'started':time.time(),'dataset_hash':digest})
        test=read(data/'test.jsonl');results={}
        for stage in manifest['stages']:
            results[stage['name']]=ev(panel([r for r in test if r['stage']==stage['index']],32),'sealed test: '+stage['name'])
        accepted=all(not failed_criteria(r) for r in results.values());atomic(out/'final_test.json',{'accepted':accepted,'stages':results})
        save(len(manifest['stages']),1,accepted=accepted,terminal=None if accepted else 'Sealed test failed')
        if accepted:save_specialist(out/(tower+'.specialist'),model,meta)
        status(state='complete' if accepted else 'blocked',accepted=accepted,reason='All gates passed' if accepted else 'Sealed test failed')
    except Exception as exc:
        status(state='failed',error=str(exc));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--initial-checkpoint',required=True)
    p.add_argument('--normal-rounds',type=int,default=8);p.add_argument('--remediation-rounds',type=int,default=6)
    p.add_argument('--attempts',type=int,default=3);p.add_argument('--examples',type=int,default=2048);p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--validation-examples',type=int,default=12);p.add_argument('--retention-examples',type=int,default=4)
    run(p.parse_args())
