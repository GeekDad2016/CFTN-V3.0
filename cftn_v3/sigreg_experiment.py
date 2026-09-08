"""Sequential paired ten-round trial; never replaces the production checkpoint.

SIGReg characteristic-function statistic follows LeJEPA MINIMAL.md (authors'
reference). Adaptation: one causal prompt-SEP embedding per maths example,
no learned projector, additive coefficient 1e-4, dedicated projection RNG.
"""
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
from .criterion_curriculum_training import batches,evaluate
from .criterion_repair import RepairController
from .criterion_sampling import balanced_panel
from .local_math_training import atomic,batch_loss,read
from .local_specialist import load_specialist,save_specialist,MathTokenizer
from .file_io import StatusPublisher

def sigreg(features,seed,projections=256):
    # A local generator does not disturb the shared dropout RNG sequence.
    with torch.autocast(features.device.type,enabled=False):
        z=features.float();g=torch.Generator(device=z.device);g.manual_seed(seed)
        directions=torch.randn(z.shape[-1],projections,device=z.device,generator=g)
        directions=directions/directions.norm(dim=0,keepdim=True).clamp_min(1e-8)
        t=torch.linspace(0,3,17,device=z.device);phi=torch.exp(-t.square()/2)
        weights=torch.full_like(t,6/16);weights[0]=weights[-1]=3/16
        angles=(z@directions).unsqueeze(-1)*t
        discrepancy=(angles.cos().mean(0)-phi).square()+angles.sin().mean(0).square()
        return ((discrepancy*(weights*phi)).sum(-1)*len(z)).mean()

def regularized_loss(model,rows,seed,coefficient):
    captured=[]
    def capture(module,args,output):
        positions=torch.tensor([len(MathTokenizer().prefix(r['prompt']))-1 for r in rows],device=output.device)
        captured.append(output[torch.arange(len(rows),device=output.device),positions])
    handle=model.final_norm.register_forward_hook(capture) if coefficient else None
    try:ce=batch_loss(model,rows)
    finally:
        if handle:handle.remove()
    penalty=sigreg(captured[0],seed) if coefficient else ce.new_zeros(())
    return ce+coefficient*penalty,ce,penalty

def run(root):
    root=Path(root);spec=json.loads((root/'experiment.json').read_text());pipeline=root.parent
    if file_hash(root/'start.specialist')!=spec['checkpoint_sha256']:raise ValueError('Protected checkpoint changed')
    data=Path(spec['data']);manifest=verify_manifest(data)
    if file_hash(data/'manifest.json')!=spec['dataset_sha256']:raise ValueError('Dataset changed')
    lock=pipeline/'native_training.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,canonical({'pid':os.getpid(),'output':str(root)}).encode());os.close(fd)
    try:
        torch.set_num_threads(4)
        train=read(data/'train.jsonl');dev=read(data/'validation.jsonl');index=spec['stage_index'];phase=manifest['stages'][index]['name']
        active=[r for r in train if r['stage']==index];prior=[r for r in train if r['stage']<index]
        quick=balanced_panel([r for r in dev if r['stage']==index],32);retention=balanced_panel([r for r in dev if r['stage']<index],8)
        schedule=RepairController();arm_results={}
        for arm,coefficient in [('baseline',0.),('sigreg',spec['coefficient'])]:
            out=root/arm;out.mkdir(exist_ok=True)
            if (out/'endpoint.json').exists():
                arm_results[arm]=json.loads((out/'endpoint.json').read_text());continue
            current=out/'current.specialist';model,saved=load_specialist(current if current.exists() else root/'start.specialist')
            model.to('cuda');optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=spec['lr'])
            optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            resume=saved['metadata'] if current.exists() else {};first=resume.get('experiment_round',1);cursor=resume.get('cursor',0)
            if current.exists() and resume.get('experiment_sha256')!=file_hash(root/'experiment.json'):raise ValueError('Experiment policy changed')
            publisher=StatusPublisher(out/'status.json');state={};started=time.time()
            def status(**values):
                state.update(values);publisher({'pid':os.getpid(),'updated':time.time(),'elapsed_seconds':time.time()-started,**state})
            def save(round_,cursor_=0):
                save_specialist(current,model,{'tower':'math','accepted':False,'experiment_arm':arm,'experiment_round':round_,
                    'cursor':cursor_,'experiment_sha256':file_hash(root/'experiment.json'),'dataset_hash':spec['dataset_sha256']},optimizer)
            def ev(rows,label):
                status(state='evaluating',evaluation=label,evaluation_done=0,evaluation_total=len(rows))
                return evaluate(model,rows,lambda done,total:status(evaluation_done=done,evaluation_total=total))
            atomic(out/'curriculum.json',manifest)
            atomic(pipeline/'queue.json',{'state':'running','current':'math','output':str(out),'pid':os.getpid(),
                'reason':'Paired SIGReg experiment; production curriculum is paused','queue':[]})
            status(state='starting',phase=phase,stage_index=index,stage_count=len(manifest['stages']),scope=manifest['stages'][index]['scope'],
                epoch=first,epochs=spec['rounds'],experiment_arm=arm,experiment_rounds=spec['rounds'],sigreg_coefficient=coefficient,
                training_mode='normal',gpu=torch.cuda.get_device_name(),checkpoint=str(current),
                policy={'examples':spec['examples'],'lr':spec['lr']},reason='Controlled experiment; no automatic stage promotion',
                completed=spec['completed'],source=str(root/'start.specialist'),parameters=sum(p.numel() for p in model.parameters()))
            for round_ in range(first,spec['rounds']+1):
                seed=9307+index*100000+spec['seed_start_round']+round_-1
                rows=schedule.rows(active,prior,spec['examples'],seed);random.Random(seed).shuffle(rows)
                chunks=list(batches(rows));ids=[r['semantic_id'] for r in rows]
                atomic(out/f'batches_{round_:02d}.json',{'seed':seed,'semantic_ids':ids})
                if arm=='sigreg' and file_hash(out/f'batches_{round_:02d}.json')!=file_hash(root/'baseline'/f'batches_{round_:02d}.json'):
                    raise ValueError('Paired batches diverged')
                status(state='training',epoch=round_,step=cursor,steps=len(chunks),evaluation=None,
                    batch_criteria=dict(collections.Counter(r['criterion'] for r in rows)))
                model.train();losses=[]
                for step,chunk in enumerate(chunks):
                    if step<cursor:continue
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss,ce,reg=regularized_loss(model,chunk,seed*10000+step,coefficient)
                    if not torch.isfinite(loss):raise RuntimeError('Non-finite experiment loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(ce.detach()))
                    if step%10==0:status(step=step+1,loss=float(ce.detach()),sigreg_loss=float(reg.detach()),total_loss=float(loss.detach()))
                    if (step+1)%100==0 or (root/'STOP').exists():save(round_,step+1)
                    if (root/'STOP').exists():status(state='paused',reason='Experiment stop requested; cursor saved');return
                cursor=0
                a=ev(quick,'routine active validation');r=ev(retention,'routine retention')
                atomic(out/f'{phase}_epoch_{round_:03d}.json',{'phase':phase,'epoch':round_,'training_mode':'normal',
                    'experiment_arm':arm,'active':a,'retention':r,'loss':sum(losses)/len(losses) if losses else None,'passed':False})
                save(round_+1)
            # Immutable endpoint weights are saved before potentially lengthy evaluation.
            save_specialist(out/'endpoint.specialist',model,{'tower':'math','accepted':False,'experiment_arm':arm,'dataset_hash':spec['dataset_sha256']},optimizer)
            a=ev(balanced_panel([r for r in dev if r['stage']==index],100000),'full experiment stage evaluation')
            r=ev(balanced_panel([r for r in dev if r['stage']<index],100000),'full experiment retention evaluation')
            result={'arm':arm,'rounds':spec['rounds'],'coefficient':coefficient,'active':a,'retention':r}
            atomic(out/f'{phase}_promotion_validation.json',{'epoch':spec['rounds'],'passed':False,'active':a,'retention':r})
            atomic(out/'endpoint.json',result);arm_results[arm]=result
            status(state='complete',accepted=False,reason='Experiment arm finished; this is not curriculum acceptance')
            del optimizer,model,saved;torch.cuda.empty_cache()
        atomic(root/'comparison.json',{'completed':time.time(),'arms':{k:{'active':{x:y for x,y in v['active'].items() if x!='samples'},
            'retention':{x:y for x,y in v['retention'].items() if x!='samples'}} for k,v in arm_results.items()},
            'note':'Matched checkpoint, optimizer, data and dropout seeds; one seed and one coefficient. No automatic winner deployment.'})
        atomic(pipeline/'queue.json',{'state':'experiment_complete','current':'math','output':str(root/'sigreg'),
            'reason':'Baseline and SIGReg endpoints saved; compare results before continuing curriculum','queue':[]})
    except Exception as exc:
        atomic(root/'error.json',{'error':str(exc),'updated':time.time()})
        raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
