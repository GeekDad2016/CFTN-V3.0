"""Bounded paired SIGReg ablation; production checkpoint is never modified."""
import json,random,shutil,time,copy,os
from pathlib import Path
import torch
from .local_specialist import load_specialist,save_specialist
from .local_math_training import read,atomic
from .criterion_curriculum_training import batches,evaluate
from .criterion_sampling import balanced_panel
from .stage_first_repair import StageFirstController
from .sigreg import regularized_loss
from .data import file_hash
from .full_curriculum_data import verify_manifest
from .file_io import StatusPublisher

def run():
    pipeline=Path('G:/ctfn-text/artifacts/v3_1');production=pipeline/'math'
    assert json.loads((production/'status.json').read_text())['state']=='paused'
    root=pipeline/'sigreg_probe_round_991';root.mkdir(exist_ok=False)
    shutil.copy2(production/'current.specialist',root/'start.specialist')
    data=Path('G:/ctfn-text/data/v3_1_foundation_v3');manifest=verify_manifest(data)
    train=read(data/'train.jsonl');dev=read(data/'validation.jsonl')
    _,initial=load_specialist(root/'start.specialist');meta=initial['metadata'];idx=meta['stage_index'];phase=manifest['stages'][idx]['name']
    assert meta['dataset_hash']==file_hash(data/'manifest.json')
    active=[r for r in train if r['stage']==idx];prior=[r for r in train if r['stage']<idx]
    panel=balanced_panel([r for r in dev if r['stage']<=idx],100000)
    policy=meta['policy'];controller=StageFirstController(state=copy.deepcopy(meta['controller']))
    schedule=[]
    for offset in range(3):
        seed=9307+idx*100000+meta['round']+offset
        rows=controller.rows(active,prior,policy['examples'],seed);random.Random(seed).shuffle(rows)
        schedule.append((seed,list(batches(rows))))
    atomic(root/'experiment.json',{'checkpoint_sha256':file_hash(root/'start.specialist'),'dataset_sha256':file_hash(data/'manifest.json'),'rounds':3,'examples_per_round':policy['examples'],'coefficients':[0,.0003],'evaluation_examples':len(panel),'training_ids':[[r['semantic_id'] for chunk in chunks for r in chunk] for _,chunks in schedule]})
    lock=pipeline/'native_training.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'output':str(root)}).encode());os.close(fd)
    torch.set_num_threads(4);results={}
    try:
        for arm,coefficient in [('baseline',0.),('sigreg',.0003)]:
            out=root/arm;out.mkdir();model,saved=load_specialist(root/'start.specialist');model.to('cuda')
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=policy['lr']);optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            publisher=StatusPublisher(out/'status.json');state={}
            def status(**values):state.update(values);publisher({'pid':os.getpid(),'updated':time.time(),**state})
            atomic(out/'curriculum.json',manifest)
            atomic(pipeline/'queue.json',{'state':'running','current':'math','output':str(out),'queue':[]})
            status(state='training',phase=phase,stage_index=idx,epoch=1,experiment_arm=arm,experiment_rounds=3,epochs=3,sigreg_coefficient=coefficient,policy={'examples':policy['examples'],'lr':policy['lr']},reason='Three-round SIGReg diagnostic; production weights preserved',completed=meta['completed'])
            means=[]
            for n,(seed,chunks) in enumerate(schedule,1):
                model.train();losses=[];penalties=[];status(state='training',epoch=n,steps=len(chunks),evaluation=None)
                for step,chunk in enumerate(chunks):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss,ce,reg=regularized_loss(model,chunk,seed*10000+step,coefficient)
                    assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    losses.append(float(ce.detach()));penalties.append(float(reg.detach()))
                    if step%10==0:status(step=step+1,loss=losses[-1],sigreg_loss=penalties[-1],total_loss=float(loss.detach()))
                means.append({'ce':sum(losses)/len(losses),'raw_sigreg':sum(penalties)/len(penalties)})
            save_specialist(out/'endpoint.specialist',model,{'tower':'math','accepted':False,'experiment_arm':arm},optimizer)
            status(state='evaluating',evaluation='full matched stage panel',evaluation_total=len(panel),evaluation_done=0)
            report=evaluate(model,panel,lambda done,total:status(evaluation_done=done,evaluation_total=total))
            results[arm]={'coefficient':coefficient,'losses':means,'evaluation':report}
            atomic(out/'result.json',results[arm]);atomic(out/f'{phase}_promotion_validation.json',{'epoch':3,'passed':False,'active':report,'retention':{}})
            status(state='complete');print(arm,report['accuracy'],flush=True)
            del model,optimizer,saved;torch.cuda.empty_cache()
        assert file_hash(root/'start.specialist')==file_hash(production/'current.specialist')
        atomic(root/'comparison.json',results)
        print('Comparison saved',root,flush=True)
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':run()
