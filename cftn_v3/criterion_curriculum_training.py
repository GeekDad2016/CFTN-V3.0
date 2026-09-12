"""Local-only native training with gated stages and bounded automatic remediation."""
from .validation_schedule import validation_pending, unlock_validation
import itertools
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
from .file_io import StatusPublisher
from .criterion_sampling import balanced_panel as panel
from .criterion_repair import failures, add_strata
from .sigreg import regularized_loss
from .stage_first_repair import StageFirstController as RepairController

from .full_curriculum_training import evaluate as raw_evaluate

def evaluate(model, rows, progress=None):
    return add_strata(raw_evaluate(model, rows, progress), rows)

failed_criteria=failures

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

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);data=Path(args.data)
    # One lock shared by every local native specialist in this pipeline.
    lock=out.parent/'native_training.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,canonical({'pid':os.getpid(),'output':str(out)}).encode());os.close(fd)
    current={};started=time.time();publisher=StatusPublisher(out/'status.json')
    def status(**kw):
        current.update(kw);publisher({'pid':os.getpid(),'updated':time.time(),'elapsed_seconds':time.time()-started,**current})
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for the local training run')
        manifest=verify_manifest(data);digest=file_hash(data/'manifest.json');atomic(out/'curriculum.json',manifest)
        train=read(data/'train.jsonl');dev=read(data/'validation.jsonl')
        latest=out/'current.specialist';model,saved=load_specialist(latest if latest.exists() else args.initial_checkpoint)
        tower=saved['tower'];resumed=latest.exists();inherited=not resumed and args.inherit_progress
        if inherited and manifest.get('parent_manifest_sha256')!=saved['metadata'].get('dataset_hash'):
            raise ValueError('Inherited checkpoint is not bound to the parent dataset')
        meta=dict(saved['metadata']) if resumed or inherited else {}
        if inherited:
            meta={k:v for k,v in meta.items() if k in ('stage_index','completed','weak','retention_weak')}
            meta['round']=1
        policy={k:getattr(args,k) for k in ('normal_rounds','remediation_rounds','attempts','examples','lr','consolidation_rounds')}
        policy.update(strict_first_stages=getattr(args,'strict_first_stages',0))
        policy.update(unbounded_loss_warmup=getattr(args,'unbounded_loss_warmup',0))
        policy.update(validation_loss_threshold=getattr(args,'validation_loss_threshold',0.))
        policy.update(validation_warmup_rounds=getattr(args,'validation_warmup_rounds',0))
        policy.update(sigreg_coefficient=getattr(args,'sigreg_coefficient',0.))
        policy.update(stage_rounds=getattr(args,'stage_rounds',240),full_check_every=getattr(args,'full_check_every',20))
        policy.update(validation_examples=getattr(args,'validation_examples',12),retention_examples=getattr(args,'retention_examples',4))
        saved_policy={'strict_first_stages':0,'unbounded_loss_warmup':0,'validation_loss_threshold':0.,'validation_warmup_rounds':0,'sigreg_coefficient':0.,'validation_examples':12,'retention_examples':4,**meta.get('policy',{})}
        upgrading=resumed and meta.get('controller_version')==2 and getattr(args,'stage_first',False)
        if upgrading:
            if meta.get('dataset_hash')!=digest or any(saved_policy.get(k)!=policy[k] for k in ('examples','lr','validation_examples','retention_examples')):
                raise ValueError('Stage-first migration must preserve data and optimizer settings')
            if (out/'sealed_test_started.json').exists() or meta.get('terminal'):
                raise ValueError('Cannot migrate a terminal or sealed run')
            if meta.get('cursor',0):raise ValueError('Stage-first migration requires a round-boundary checkpoint')
            import shutil
            backup=out/'before_stage_first.specialist'
            if not backup.exists():shutil.copy2(latest,backup)
            old_round=meta['round'];idx=meta['stage_index'];phase=manifest['stages'][idx]['name']
            history=[json.loads(p.read_text()) for p in out.glob(phase+'_epoch_*.json')]
            normal_count=sum(r.get('training_mode')!='repair' for r in history if r['epoch']<old_round)
            repair_count=sum(r.get('training_mode')=='repair' for r in history if r['epoch']<old_round)
            fresh=RepairController(args.normal_rounds,args.remediation_rounds,args.attempts,args.consolidation_rounds)
            fresh.state.update(normal_done=normal_count,normal_total=normal_count,recovery_total=repair_count,legacy_recovery=repair_count)
            meta['controller']=fresh.state
            atomic(out/'stage_first_migration.json',{'resume_round':old_round,'normal_completed':normal_count,'recovery_completed':repair_count,'old_policy':saved_policy,'new_policy':policy,'backup':str(backup)})
        elif resumed and saved_policy!=policy and meta.get('dataset_hash')==digest and all(saved_policy.get(k)==v for k,v in policy.items() if k!='strict_first_stages'):
            import shutil
            backup=out/'before_strict_foundation_gates.specialist'
            if not backup.exists():shutil.copy2(latest,backup)
            meta['controller'].update(streak=0,full_streak=0,full_pass_round=None)
            meta['consecutive']=0;upgrading=True
        elif resumed and getattr(args,'upgrade_validation_warmup',False) and saved_policy!=policy:
            if meta.get('dataset_hash')!=digest or meta.get('controller_version')!=3 or any(saved_policy.get(k)!=v for k,v in policy.items() if k not in ('validation_warmup_rounds','validation_loss_threshold','unbounded_loss_warmup')):
                raise ValueError('Warmup migration may only change validation scheduling')
            import shutil
            backup=out/('before_unbounded_loss_warmup.specialist' if policy['unbounded_loss_warmup'] else 'before_validation_loss_threshold.specialist' if policy['validation_loss_threshold'] else 'before_validation_warmup.specialist')
            if not backup.exists():shutil.copy2(latest,backup)
            meta['controller'].update(streak=0,full_streak=0,full_pass_round=None)
            meta['controller']['validation_enabled']=False
            if policy['unbounded_loss_warmup']:
                meta['controller'].update(mode='normal',focus=None,normal_done=0,recovery_blocks=0,attempt_counts={},repair_done=0)
                meta['cursor']=0
            meta['consecutive']=0;upgrading=True
        elif resumed and (meta.get('dataset_hash')!=digest or saved_policy!=policy or meta.get('controller_version')!=3):raise ValueError('Resume data or policy changed')
        if meta.get('accepted'):status(state='complete',accepted=True);return
        if meta.get('terminal'):status(state='blocked',reason=meta['terminal']);return
        model.to('cuda');torch.manual_seed(9307);torch.set_num_threads(4)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
        if (resumed or inherited) and saved.get('optimizer'):
            optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        completed=meta.get('completed',[]);start_stage=meta.get('stage_index',0);start_round=meta.get('round',1)
        cursor=meta.get('cursor',0);consecutive=meta.get('consecutive',0);weak=meta.get('weak',[]);retention_weak=meta.get('retention_weak',[])
        maxround=args.normal_rounds*(args.attempts+1)+args.remediation_rounds*args.attempts+meta.get('controller',{}).get('legacy_recovery',0)
        if maxround<1 or policy['full_check_every']<1:raise ValueError('Round budgets must be positive')
        controller=RepairController(args.normal_rounds,args.remediation_rounds,args.attempts,args.consolidation_rounds,meta.get('controller'))
        status(state='starting',phase='curriculum validation',source=meta.get('adopted_sigreg_experiment',{}).get('endpoint',str(args.initial_checkpoint)),dataset=str(data),
            parameters=sum(p.numel() for p in model.parameters()),gpu=torch.cuda.get_device_name(),
            checkpoint=str(latest),completed=completed,epochs=maxround,policy=policy,
            resumed=resumed,resume_round=start_round,resume_cursor=cursor,
            inherited_progress=inherited,controller_version=3,
            queue='String is waiting for Maths acceptance and GPU release')
        def save(index,round_,cursor_=0,**extra):
            nonlocal meta
            meta={'tower':tower,'accepted':False,'dataset_hash':digest,'policy':policy,'stage_index':index,'round':round_,
                'cursor':cursor_,'completed':completed,'consecutive':consecutive,'weak':weak,'retention_weak':retention_weak,
                'source_checkpoint':str(args.initial_checkpoint),'controller':controller.state,'controller_version':3,
                'adopted_sigreg_experiment':meta.get('adopted_sigreg_experiment',{}),**extra}
            save_specialist(latest,model,meta,optimizer)
        def ev(rows,label):
            status(state='evaluating',evaluation=label,evaluation_done=0,evaluation_total=len(rows))
            return evaluate(model,rows,lambda done,total:status(evaluation_done=done,evaluation_total=total))
        for index in range(start_stage,len(manifest['stages'])):
            strict_gate=index<policy['strict_first_stages']
            def failed_criteria(report,retention=False,baseline=None):return failures(report,retention,baseline,strict=strict_gate)
            stage=manifest['stages'][index];phase=stage['name'];active=[r for r in train if r['stage']==index]
            prior=[r for r in train if r['stage']<index];validation=[r for r in dev if r['stage']==index]
            active_panel=panel(validation,policy['validation_examples']);retention_panel=panel([r for r in dev if r['stage']<index],policy['retention_examples'])
            controller=RepairController(args.normal_rounds,args.remediation_rounds,args.attempts,args.consolidation_rounds,
                meta.get('controller') if index==start_stage else None)
            entry_path=out/f'{phase}_before.json'
            status(phase=phase,stage_index=index,strict_gate=strict_gate,scope=stage['scope'],epoch=start_round if index==start_stage else 1,
                stage_count=len(manifest['stages']),active_examples=len(active),completed=completed)
            if entry_path.exists():entry=json.loads(entry_path.read_text())
            elif validation_pending(controller.state,policy):
                entry={'baseline_deferred':True,'active':{'criteria':{}},'retention':{'criteria':{},'accuracy':None}}
                atomic(entry_path,entry)
            else:
                entry={'active':ev(active_panel,'stage baseline'),'retention':ev(retention_panel,'retention baseline')}
                atomic(entry_path,entry)
            if inherited and index==start_stage:
                weak=failed_criteria(entry['active']);retention_weak=failed_criteria(entry['retention'],True)
                inherited=False
                save(index,start_round)
            if upgrading and index==start_stage:save(index,start_round,cursor)
            for round_ in itertools.count(start_round if index==start_stage else 1):
                begun=time.time();mode=controller.state['mode'];focus=controller.state['focus']
                warmup=validation_pending(controller.state,policy)
                if policy['unbounded_loss_warmup']:
                    maxround=None if warmup else (round_ if controller.state.get('recovery_queue') else controller.state['validation_start_round'])+args.normal_rounds*(args.attempts+1)+args.remediation_rounds*args.attempts-1
                if maxround is not None and round_>maxround:
                    save(index,round_,terminal='Stage round budget exhausted without two consecutive full passes')
                    status(state='blocked',reason='Stage round budget exhausted without two consecutive full passes; no stage skipped');return
                attempt=controller.state['attempt_counts'].get(focus,0)
                status(epoch=round_,epochs=maxround,recovery_queue=controller.state.get('recovery_queue'),remediation_attempt=attempt,remediation_criteria=[focus] if mode=='repair' else weak,
                    training_mode=mode,focused_criterion=focus if mode=='repair' else None,
                    consolidation_done=controller.state['consolidation_done'],step=cursor,
                    normal_done=controller.state['normal_done'],normal_total=controller.state.get('normal_total',0),
                    recovery_total=controller.state.get('recovery_total',0),recovery_blocks=controller.state.get('recovery_blocks',0),
                    validation_suppressed=warmup,validation_loss_threshold=policy['validation_loss_threshold'],round_mean_loss=controller.state.get('round_mean_loss'),
                    validation_resumes_normal_round=None if policy['validation_loss_threshold'] else policy['validation_warmup_rounds']+1,
                    next_routine_check=None if warmup and policy['validation_loss_threshold'] else round_+policy['validation_warmup_rounds']-controller.state.get('normal_total',controller.state['normal_done']) if warmup else round_,
                    next_full_check=None if warmup else controller.next_check(round_,policy['full_check_every'],maxround),
                    full_consecutive=controller.state.get('full_streak',0))
                seed=9307+index*100000+round_
                # Active training is 75%, prior accepted skills 25%; stage zero has no replay.
                rows=controller.rows(active,prior,args.examples,seed)
                random.Random(seed).shuffle(rows)
                status(batch_criteria=dict(collections.Counter(r['criterion'] for r in rows)),
                    batch_decisions=dict(collections.Counter(r['answer'] for r in rows)) if mode=='repair' else {})
                chunks=list(batches(rows));losses=[];model.train()
                if not cursor:controller.state.update(round_loss_sum=0.,round_loss_count=0)
                complete_loss=not cursor or controller.state.get('round_loss_count',0)==cursor
                status(state='remediating' if mode=='repair' else 'training',steps=len(chunks),evaluation=None)
                for step,chunk in enumerate(chunks):
                    if step<cursor:continue
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        if policy['sigreg_coefficient']:
                            loss,ce,reg=regularized_loss(model,chunk,seed*10000+step,policy['sigreg_coefficient'])
                        else:
                            loss=batch_loss(model,chunk);ce=loss;reg=loss.new_zeros(())
                    if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(ce.detach()))
                    controller.state['round_loss_sum']=controller.state.get('round_loss_sum',0.)+losses[-1]
                    controller.state['round_loss_count']=controller.state.get('round_loss_count',0)+1
                    if step%10==0:status(step=step+1,loss=losses[-1],total_loss=float(loss.detach()),sigreg_loss=float(reg.detach()),sigreg_coefficient=policy['sigreg_coefficient'],gpu_bytes=torch.cuda.memory_allocated())
                    if (step+1)%100==0 or (out/'STOP').exists():save(index,round_,step+1)
                    if (out/'STOP').exists():status(state='paused',reason='Safe stop requested; optimizer and cursor saved');return
                cursor=0
                mean_loss=controller.state['round_loss_sum']/controller.state['round_loss_count'] if controller.state.get('round_loss_count') else None
                controller.state['round_mean_loss']=mean_loss if complete_loss else None
                if warmup and policy['validation_loss_threshold'] and unlock_validation(controller.state,policy,mean_loss if complete_loss else None):
                    warmup=False
                    if policy['unbounded_loss_warmup']:
                        controller.state['validation_start_round']=round_
                        maxround=round_+args.normal_rounds*(args.attempts+1)+args.remediation_rounds*args.attempts-1
                    status(validation_suppressed=False,reason=controller.state['validation_unlock_reason'])
                status(round_mean_loss=controller.state['round_mean_loss'])
                request=out/'VALIDATE_REQUEST.json'
                if request.exists():
                    status(manual_validation='running')
                    # A diagnostic check does not advance the controller or consume RNG.
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        diagnostic_active=ev(active_panel,'manual active validation')
                        diagnostic_retention=ev(retention_panel,'manual prior-stage retention')
                    diagnostic={'phase':phase,'epoch':round_,'loss':controller.state.get('round_mean_loss'),
                        'active':diagnostic_active,'retention':diagnostic_retention,'updated':time.time(),
                        'passed':not failed_criteria(diagnostic_active) and not failed_criteria(diagnostic_retention,True,entry['retention']['criteria'])}
                    atomic(out/f'{phase}_manual_round_{round_:06d}.json',diagnostic)
                    atomic(out/f'{phase}_manual_validation.json',diagnostic)
                    request.unlink(missing_ok=True)
                    status(manual_validation='complete',evaluation=None,state='training')

                if warmup:
                    controller.state['normal_done']+=1
                    controller.state['normal_total']=controller.state.get('normal_total',0)+1
                    controller.state.update(streak=0,full_streak=0,full_pass_round=None)
                    consecutive=0
                    atomic(out/f'{phase}_training_only_{round_:03d}.json',{'phase':phase,'epoch':round_,'validation_skipped':True,
                        'normal_total':controller.state['normal_total'],'loss':sum(losses)/len(losses) if losses else current.get('loss')})
                    save(index,round_+1)
                    status(state='training',normal_done=controller.state['normal_done'],normal_total=controller.state['normal_total'],
                        reason=('Training until round-average CE loss <= '+str(policy['validation_loss_threshold'])+('; no round limit before threshold' if policy['unbounded_loss_warmup'] else '; budget-end evaluation remains required')) if policy['validation_loss_threshold'] else 'Initial normal training; validation starts at normal round '+str(policy['validation_warmup_rounds']+1),passed=None)
                    continue
                status(reason=None)
                observed=ev(active_panel,'active validation');retained=ev(retention_panel,'prior-stage retention')
                weak=failed_criteria(observed);retention_weak=failed_criteria(retained,True,entry['retention']['criteria'])
                passed=not weak and not retention_weak
                try:gate=controller.observe(weak,retention_weak)
                except RuntimeError as exc:
                    save(index,round_+1,terminal=str(exc));status(state='blocked',reason=str(exc));return
                consecutive=controller.state['streak']
                report={'phase':phase,'epoch':round_,'loss':sum(losses)/len(losses) if losses else current.get('loss'),
                    'training_mode':mode,'focused_criterion':focus if mode=='repair' else None,
                    'controller':dict(controller.state),'active':observed,'retention':retained,'retention_baseline':entry['retention']['accuracy'],
                    'passed':passed,'consecutive':consecutive,'remediation_attempt':attempt,'failed_criteria':weak,
                    'retention_failed_criteria':retention_weak,'elapsed':time.time()-begun}
                atomic(out/f'{phase}_epoch_{round_:03d}.json',report)
                save(index,round_+1);status(state='evaluated',passed=passed,accuracy=observed['accuracy'],retention=retained['accuracy'],consolidation_done=controller.state['consolidation_done'])
                if controller.state.get('recovery_queue') and (controller.state['repair_done']%5==0 or controller.state.get('recovery_gate_streak',0)>0):
                    if controller.state.get('subskill_recovery'):
                        from .subskill_recovery import matches
                        focused_rows=panel([r for r in validation if matches(r,controller.state['focus'])],100000)
                    else:
                        focused_rows=panel([r for r in validation if r['criterion']==controller.state['focus']],100000)
                    if not focused_rows:raise ValueError('Focused recovery has no held-out questions')
                    focused_report=ev(focused_rows,'complete focused recovery validation')
                    status(recovery_check_criterion=controller.state['focus'],recovery_check_round=round_,recovery_check_accuracy=focused_report['accuracy'],recovery_check_examples=len(focused_rows),recovery_check_passed=not failed_criteria(focused_report))
                    atomic(out/f'{phase}_recovery_gate_{round_:06d}.json',{'phase':phase,'epoch':round_,'criterion':controller.state['focus'],'active':focused_report})
                    try:
                        focused_fail=failed_criteria(focused_report)
                        controller.recovery_result([controller.state['focus']] if focused_fail else [])
                        if not controller.state.get('recovery_queue'):controller.state['validation_start_round']=round_+1
                    except RuntimeError as exc:
                        save(index,round_+1,terminal=str(exc));status(state='blocked',reason=str(exc));return
                    save(index,round_+1)
                    status(recovery_queue=controller.state.get('recovery_queue'),recovery_gate_streak=controller.state.get('recovery_gate_streak'),training_mode=controller.state['mode'])
                if controller.due(round_,policy['full_check_every'],maxround):
                    full=ev(panel(validation,100000),'complete stage validation')
                    cumulative=ev(panel([r for r in dev if r['stage']<index],100000),'complete retention gate')
                    full_fail=failed_criteria(full);cum_fail=failed_criteria(cumulative,True)
                    if controller.state.get('subskill_recovery'):
                        from .subskill_recovery import failed_subskills
                        controller.state['pending_subskills']=failed_subskills(full,panel(validation,100000))
                    try:promoted=controller.full_result(round_,full_fail,cum_fail)
                    except RuntimeError as exc:
                        save(index,round_+1,terminal=str(exc));status(state='blocked',reason=str(exc));return
                    atomic(out/f'{phase}_promotion_validation.json',{'phase':phase,'epoch':round_,'updated':time.time(),'active':full,'retention':cumulative,'passed':not full_fail and not cum_fail,'full_consecutive':controller.state.get('full_streak',0),'promoted':promoted})
                    status(promotion_passed=not full_fail and not cum_fail,promotion_failed_criteria=full_fail,
                        promotion_retention_failed_criteria=cum_fail)
                    if promoted:
                        completed.append(phase);consecutive=0;weak=[];retention_weak=[]
                        controller=RepairController(args.normal_rounds,args.remediation_rounds,args.attempts,args.consolidation_rounds)
                        save(index+1,1)
                        status(state='stage_complete',reason='Mastery and retention passed',completed=completed);break
                    weak=full_fail;retention_weak=cum_fail;consecutive=0
                    save(index,round_+1)
            start_round=1;cursor=0
        # Tests never influence remediation. Once consumed, a failure requires a new release evaluation.
        if (out/'sealed_test_started.json').exists():
            status(state='blocked',reason='Sealed test already opened; manual review required');return
        atomic(out/'sealed_test_started.json',{'started':time.time(),'dataset_hash':digest})
        test=read(data/'test.jsonl');results={}
        for stage in manifest['stages']:
            results[stage['name']]=ev(panel([r for r in test if r['stage']==stage['index']],32),'sealed test: '+stage['name'])
        accepted=all(not failures(results[stage['name']],strict=stage['index']<policy['strict_first_stages']) for stage in manifest['stages']);atomic(out/'final_test.json',{'accepted':accepted,'stages':results})
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
    p.add_argument('--strict-first-stages',type=int,default=0);p.add_argument('--unbounded-loss-warmup',type=int,choices=(0,1),default=0);p.add_argument('--validation-loss-threshold',type=float,default=0.);p.add_argument('--validation-warmup-rounds',type=int,default=0);p.add_argument('--upgrade-validation-warmup',action='store_true')
    p.add_argument('--sigreg-coefficient',type=float,default=0.)
    p.add_argument('--stage-first',action='store_true')
    p.add_argument('--stage-rounds',type=int,default=240);p.add_argument('--full-check-every',type=int,default=20);p.add_argument('--upgrade-recovery',action='store_true')
    p.add_argument('--inherit-progress',action='store_true');p.add_argument('--consolidation-rounds',type=int,default=3)
    p.add_argument('--validation-examples',type=int,default=12);p.add_argument('--retention-examples',type=int,default=4)
    run(p.parse_args())
