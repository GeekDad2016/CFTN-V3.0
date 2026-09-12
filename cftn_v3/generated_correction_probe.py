"""Generated training-failure mining, supervised correction trial, gated resume."""
import copy
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
import torch
from .config import canonical
from .data import file_hash
from .file_io import atomic_json, atomic_write, StatusPublisher
from .full_curriculum_data import verify_manifest
from .local_math_training import read
from .local_specialist import load_specialist, save_specialist
from .criterion_curriculum_training import batches, evaluate
from .criterion_sampling import balanced_panel
from .stage_first_repair import StageFirstController
from .sigreg import regularized_loss
from .subskill_recovery import sample, failed_subskills
from .generated_correction import verify, expected_answer, correction_rows
from .balanced_curriculum_data import comparison_family

ROOT = Path('G:/ctfn-text/artifacts/v3_1')
DATA = Path('G:/ctfn-text/data/v3_1_subskill_v5')
NEW_DATA = Path('G:/ctfn-text/data/v3_1_subskill_v5')
ROUNDS = 5

def counts(report):
    return {k:sum(not s[k] for s in report['samples'])
            for k in ('answer_correct','trace_correct','format_correct')}

def no_worse(candidate, reference):
    return all(counts(candidate[split])[k] <= counts(reference[split])[k]
               for split in ('active','retention') for k in counts(candidate[split]))

def choose(results):
    base, targeted, start = (results[k] for k in ('baseline','targeted','start'))
    improves = sum(counts(targeted['active']).values()) < sum(counts(base['active']).values())
    if improves and no_worse(targeted,base) and no_worse(targeted,start): return 'targeted'
    if no_worse(base,start): return 'baseline'
    return 'start'

def run():
    assert not (ROOT/'pipeline.lock').exists()
    lock=ROOT/'native_training.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,canonical({'pid':os.getpid(),'output':str(ROOT/'generated_correction_probe')}).encode());os.close(fd)
    production=ROOT/'math'; start=production/'current.specialist'
    root=ROOT/'generated_correction_probe'; root.mkdir(exist_ok=False)
    shutil.copy2(start,root/'start.specialist')
    original_hash=file_hash(start)
    config_path=Path('config/local_curriculum_v31.json')
    original_config=config_path.read_bytes()
    adopted=False
    try:
        manifest=verify_manifest(DATA); new_manifest=verify_manifest(NEW_DATA)
        assert file_hash(DATA/'validation.jsonl')==file_hash(NEW_DATA/'validation.jsonl')
        assert file_hash(DATA/'test.jsonl')==file_hash(NEW_DATA/'test.jsonl')
        train=read(DATA/'train.jsonl'); new_train=read(NEW_DATA/'train.jsonl'); dev=read(DATA/'validation.jsonl')
        initial=torch.load(root/'start.specialist',map_location='cpu',weights_only=True)
        meta=initial['metadata']; policy=meta['policy']; idx=meta['stage_index']
        assert idx==1 and meta['dataset_hash']==file_hash(DATA/'manifest.json')
        assert not meta.get('terminal')
        panels={'active':balanced_panel([r for r in dev if r['stage']==idx],100000),
                'retention':balanced_panel([r for r in dev if r['stage']<idx],100000)}
        active=[r for r in train if r['stage']==idx]; expanded=[r for r in new_train if r['stage']==idx]
        prior=[r for r in train if r['stage']<idx]
        atomic_json(root/'experiment.json',{'start_sha256':original_hash,'round':meta['round'],
                    'rounds_per_arm':ROUNDS,'examples_per_round':policy['examples'],
                    'old_dataset':file_hash(DATA/'manifest.json'),'new_dataset':file_hash(NEW_DATA/'manifest.json'),
                    'selection':'Targeted must improve active errors over baseline with no metric regression against baseline or start; otherwise baseline if non-regressing, else start.'})
        torch.set_num_threads(4); results={}; failed=[]
        reserved={comparison_family(r['ir']) for r in dev+read(DATA/'test.jsonl')}
        quarantine=DATA/'quarantined_heldout.jsonl'
        if quarantine.exists():
            reserved.update(comparison_family(r.get('record',r)['ir']) for r in read(quarantine))
        pool=balanced_panel(active,96)+balanced_panel([r for r in prior if r['ir']['op']=='compare'],96)
        mining=[]
        for row in pool:
            if comparison_family(row['ir']) in reserved: continue
            assert expected_answer(row['ir'])==row['answer']
            assert not verify(row,row['target'])['needs_correction']
            mining.append(row)
        assert mining
        atomic_json(root/'mining_manifest.json',{'examples':len(mining),'training_ids':[r['semantic_id'] for r in mining],
            'heldout_excluded':True,'source_dataset':file_hash(DATA/'manifest.json')})
        for arm in ('start','baseline','targeted'):
            out=root/arm;out.mkdir()
            atomic_json(out/'curriculum.json',manifest)
            atomic_json(ROOT/'queue.json',{'state':'running','current':'math','output':str(out),'queue':[]})
            publisher=StatusPublisher(out/'status.json'); state={}
            def status(**values):
                state.update(values);publisher({'pid':os.getpid(),'updated':time.time(),**state})
            status(state='starting',phase=manifest['stages'][idx]['name'],stage_index=idx,
                   experiment_arm=arm,epoch=0,epochs=ROUNDS,completed=meta['completed'],
                   reason=f'Generated correction comparison: {arm}. Production checkpoint protected.',
                   policy=policy,dataset=str(NEW_DATA if arm=='targeted' else DATA))
            model,saved=load_specialist(root/'start.specialist');model.to('cuda')
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=policy['lr'])
            optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            controller=StageFirstController(state=copy.deepcopy(meta['controller'])); means=[]; schedules=[]
            if arm!='start':
                for offset in range(ROUNDS):
                    seed=9307+idx*100000+meta['round']+offset
                    focus=('expression_decision','borrowing','repeated_digits','expression_decision','borrowing')[offset]
                    rows=(correction_rows(failed,active,prior,policy['examples'],seed) if arm=='targeted' and failed
                          else controller.rows(active,prior,policy['examples'],seed))
                    random.Random(seed).shuffle(rows); chunks=list(batches(rows))
                    schedules.append([r['semantic_id'] for r in rows])
                    status(state='training',epoch=offset+1,steps=len(chunks),step=0,
                           focused_criterion='Generated training failures' if arm=='targeted' and failed else None,evaluation=None)
                    losses=[];model.train()
                    for step,chunk in enumerate(chunks):
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            loss,ce,reg=regularized_loss(model,chunk,seed*10000+step,policy['sigreg_coefficient'])
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite experiment loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                        losses.append(float(ce.detach()))
                        if step%10==0:status(step=step+1,loss=losses[-1],sigreg_coefficient=policy['sigreg_coefficient'])
                    means.append(sum(losses)/len(losses));status(round_mean_loss=means[-1])
                    save_specialist(out/'endpoint.specialist',model,meta,optimizer)
                atomic_json(out/'training_ids.json',schedules)
            report={}
            for split,rows in panels.items():
                status(state='evaluating',evaluation=f'Full {split} comparison panel',evaluation_total=len(rows),evaluation_done=0)
                report[split]=evaluate(model,rows,lambda done,total:status(evaluation_done=done,evaluation_total=total))
            if arm=='start':
                evidence=[]
                status(state='evaluating',evaluation='Mining generated errors on training questions',evaluation_done=0,evaluation_total=len(mining))
                for number,row in enumerate(mining,1):
                    output,complete=model.generate(row['prompt'])
                    feedback=verify(row,output)
                    if feedback['needs_correction']:failed.append(row)
                    evidence.append({'semantic_id':row['semantic_id'],'prompt':row['prompt'],'expected':row['answer'],
                                     'target':row['target'],'output':output,'complete':complete,'feedback':feedback})
                    if number%8==0 or number==len(mining):
                        atomic_json(root/'mined_generations.json',evidence)
                        status(evaluation_done=number,generated_failures=len(failed))
                atomic_json(root/'correction_training_rows.json',failed)
                print('Verified training failures',len(failed),'of',len(mining),flush=True)
            report['losses']=means;results[arm]=report
            atomic_json(out/'result.json',report)
            atomic_json(out/(manifest['stages'][idx]['name']+'_promotion_validation.json'),
                        {'epoch':ROUNDS if arm!='start' else 0,'active':report['active'],'retention':report['retention'],
                         'passed':False,'promoted':False,'full_consecutive':0})
            status(state='complete',accuracy=report['active']['accuracy'],retention=report['retention']['accuracy'])
            print(arm,{s:counts(report[s]) for s in panels},flush=True)
            del model,optimizer,saved;torch.cuda.empty_cache()
        selected=choose(results)
        assert file_hash(start)==original_hash, 'Production checkpoint changed during trial'
        payload=torch.load(root/('start.specialist' if selected=='start' else selected+'/endpoint.specialist'),map_location='cpu',weights_only=True)
        adopted_meta=copy.deepcopy(meta); c=adopted_meta['controller']
        if selected!='start':
            adopted_meta['round']+=ROUNDS
            c['normal_total']=c.get('normal_total',0)+ (ROUNDS if selected=='baseline' else 0)
            c['normal_done']+=ROUNDS if selected=='baseline' else 0
        if selected=='targeted':
            adopted_meta['dataset_hash']=file_hash(NEW_DATA/'manifest.json')
            queue=failed_subskills(results[selected]['active'],panels['active'])
            c.update(subskill_recovery=True,pending_subskills=queue,recovery_queue=queue,
                     mode='repair' if queue else 'normal',focus=queue[0] if queue else None,
                     repair_done=0,recovery_gate_streak=0,normal_done=0,
                     recovery_total=c.get('recovery_total',0)+ROUNDS,
                     recovery_blocks=c.get('recovery_blocks',0)+1)
        c.update(streak=0,full_streak=0,full_pass_round=None,round_loss_sum=0.,round_loss_count=0)
        adopted_meta.update(consecutive=0,cursor=meta['cursor'] if selected=='start' else 0)
        payload['metadata']=adopted_meta
        atomic_write(start,lambda stream:torch.save(payload,stream))
        config=json.loads(original_config)
        if selected=='targeted':config['queue'][0]['data']=str(NEW_DATA)
        atomic_json(config_path,config)
        atomic_json(root/'decision.json',{'selected':selected,'resume_round':adopted_meta['round'],
                    'results':{a:{s:counts(v[s]) for s in panels} for a,v in results.items()}})
        adopted=True
    except Exception as exc:
        atomic_json(root/'error.json',{'error':repr(exc),'time':time.time()})
        # Roll back a partial adoption, preserving the protected start and original config.
        if not adopted:
            atomic_write(start,lambda stream:stream.write((root/'start.specialist').read_bytes()))
            atomic_write(config_path,lambda stream:stream.write(original_config))
        raise
    finally:
        lock.unlink(missing_ok=True)
    (production/'STOP').unlink(missing_ok=True)
    with (ROOT/'pipeline.stdout.log').open('w') as stdout,(ROOT/'pipeline.stderr.log').open('w') as stderr:
        job=subprocess.Popen([sys.executable,'-u','-m','cftn_v3.local_curriculum_pipeline','--config',str(config_path.resolve())],
                             stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    (ROOT/'pipeline.pid').write_text(str(job.pid))
    print('Selected',selected,'and resumed curriculum, PID',job.pid,flush=True)

if __name__=='__main__':run()
