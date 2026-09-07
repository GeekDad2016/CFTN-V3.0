"""Local-first specialist curriculum, with no GPT or other tower allocation."""
import argparse
import copy
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from .config import canonical,identity
from .data import file_hash
from .local_specialist import LocalMathTower,MathTokenizer,load_legacy,load_specialist,save_specialist
from .file_io import atomic_json

PHASES=('multiply','power','pythagoras','fraction_add','percent_of','divide','linear_solve')


def atomic(path,value):
    atomic_json(path,value)


def semantic(ir):
    op=ir['op']
    if op=='power':return identity([op,int(ir['base']),int(ir['exponent'])])
    if op=='multiply':return identity([op,*sorted((int(ir['left']),int(ir['right'])))])
    return identity(ir)


def row(ir,target,answer,phase):
    return {'prompt':canonical(ir),'target':target,'answer':str(answer),'phase':phase,'semantic_id':semantic(ir)}


def prepare(parent,destination):
    parent,destination=Path(parent),Path(destination);destination.mkdir(parents=True,exist_ok=True)
    manifest=destination/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        for name,digest in saved['files'].items():
            if file_hash(destination/name)!=digest:raise ValueError('Local dataset checksum mismatch')
        return saved
    source=json.loads((parent/'manifest.json').read_text())
    groups={s:[] for s in ('train','validation','test')};assigned={}
    for split in groups:
        meta=source['splits'][split];path=parent/meta['path']
        if file_hash(path)!=meta['sha256']:raise ValueError('V12 source checksum mismatch')
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);ir=r['math_ir'];sid=semantic(ir)
            if sid in assigned and assigned[sid]!=split:raise ValueError('Source semantic split conflict')
            assigned[sid]=split
            # Reuse the original typed JSON interface and compact procedural target.
            groups[split].append({'prompt':r['problem'],'target':r['target_trace'],'answer':r['answer'],
                'phase':ir['op'],'semantic_id':sid})
    additions=[]
    for a in range(145):
        for b in range(2,13):
            ir={'op':'multiply','left':a,'right':b,'type':'math_problem_v1'}
            additions.append(row(ir,f'<work>{a}*{b}={a*b}</work><answer>{a*b}</answer>',a*b,'multiply'))
    for base in range(-20,21):
        for exponent in (2,3,4):
            ir={'op':'power','base':base,'exponent':exponent,'type':'math_problem_v1'}
            value=base;steps=[]
            for _ in range(exponent-1):
                a=f'({value})' if value<0 else str(value);b=f'({base})' if base<0 else str(base)
                steps.append(f'{a}*{b}={value*base}')
                # Train the arithmetic required by the bounded powers curriculum.
                m={'op':'multiply','left':value,'right':base,'type':'math_problem_v1'}
                additions.append(row(m,f'<work>{a}*{b}={value*base}</work><answer>{value*base}</answer>',value*base,'multiply'))
                value*=base
            additions.append(row(ir,'<work>'+';'.join(steps)+f'</work><answer>{value}</answer>',value,'power'))
    # Bounded powers replace huge-power training in the active curriculum, while
    # the original sealed test set stays on disk for later non-blocking diagnostics.
    for split in groups:
        groups[split]=[r for r in groups[split] if r['phase'] not in ('multiply','power')]
    used=set()
    for r in additions:
        sid=r['semantic_id']
        if sid in used:continue
        used.add(sid)
        bucket=int(sid[:8],16)%10
        split=assigned.get(sid,'train' if bucket<8 else 'validation' if bucket==8 else 'test')
        groups[split].append(r)
    tok=MathTokenizer();files={};counts={};maximum=0
    for split,rows in groups.items():
        rows=list({r['semantic_id']:r for r in rows}.values())
        rows.sort(key=lambda r:r['semantic_id'])
        counts[split]={p:sum(r['phase']==p for r in rows) for p in PHASES}
        for r in rows:
            n=len(tok.prefix(r['prompt']))+len(tok.encode(r['target']))+1
            maximum=max(maximum,n)
            if n>2048:raise ValueError('Actual tokenizer context overflow')
        path=destination/(split+'.jsonl');path.write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
        files[path.name]=file_hash(path)
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        if {r['semantic_id'] for r in groups[a]} & {r['semantic_id'] for r in groups[b]}:raise ValueError('Split overlap')
    result={'format':'local_math_curriculum_v1','source_manifest_sha256':file_hash(parent/'manifest.json'),
        'files':files,'counts':counts,'max_training_tokens':maximum,'phases':PHASES,
        'scope':'Bounded arithmetic and school mathematics; not general-language or PhD capability'}
    atomic(manifest,result);return result


def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def batch_loss(model,rows):
    tok=MathTokenizer();encoded=[];weighted=[]
    for r in rows:
        prefix=tok.prefix(r['prompt']);target=tok.encode(r['target'])+[2]
        weights=[1.]*len(target)
        # Give computed RHS and final answers more weight than copied operators.
        for match in re.finditer(r'=(-?\d+(?:/\d+)?)',r['target']):
            for i in range(match.start(1),match.end(1)):weights[i]=3.
        m=re.search(r'<answer>(.*?)</answer>',r['target'])
        if m:
            for i in range(m.start(1),m.end(1)):weights[i]=4.
        encoded.append(prefix+target);weighted.append([0.]*len(prefix)+weights)
    length=max(map(len,encoded));device=next(model.parameters()).device
    ids=torch.zeros(len(rows),length,dtype=torch.long,device=device);weights=torch.zeros_like(ids,dtype=torch.float)
    for i,(values,w) in enumerate(zip(encoded,weighted)):
        ids[i,:len(values)]=torch.tensor(values,device=device);weights[i,:len(w)]=torch.tensor(w,device=device)
    logits=model(ids[:,:-1]);loss=F.cross_entropy(logits.float().reshape(-1,260),ids[:,1:].reshape(-1),reduction='none').reshape(ids.shape[0],-1)
    return (loss*weights[:,1:]).sum()/weights[:,1:].sum()


def evaluate(model,rows):
    samples=[];model.eval()
    for r in rows:
        # The byte budget is derived from the complete gold trace, never its value.
        output,eos=model.generate(r['prompt'],min(model.context-len(MathTokenizer().prefix(r['prompt'])),len(r['target'].encode())+64))
        matches=re.findall(r'<answer>(.*?)</answer>',output)
        answer=matches[-1].strip() if len(matches)==1 else None
        samples.append({'prompt':r['prompt'],'expected':r['answer'],'expected_trace':r['target'],'output':output,
            'answer_correct':answer==r['answer'],'complete':eos,
            'trace_correct':eos and ''.join(output.split())==''.join(r['target'].split()),'phase':r['phase']})
    n=max(1,len(samples))
    return {'examples':len(samples),'accuracy':sum(s['answer_correct'] for s in samples)/n,
        'trace_accuracy':sum(s['trace_correct'] for s in samples)/n,'samples':samples}


def balanced(rows,count,seed):
    rng=random.Random(seed);groups=defaultdict(list)
    for r in rows:groups[r['phase']].append(r)
    keys=sorted(groups)
    if not keys:return []
    return [rng.choice(groups[keys[i%len(keys)]]) for i in range(count)]


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    lock=out/'training.lock'
    import os
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    def status(**values):
        atomic(out/'status.json',{'pid':os.getpid(),'updated':time.time(),**values})
        print(canonical(values),flush=True)
    try:
        manifest=prepare(args.data,Path(args.dataset_output))
        data=Path(args.dataset_output);train=read(data/'train.jsonl');dev=read(data/'validation.jsonl');test=read(data/'test.jsonl')
        config=json.loads(Path(args.config).read_text());spec=config['math_tower']
        latest=out/'current.specialist';state=None;completed=[];phase_index=0;start_epoch=0
        if latest.exists():
            model,state=load_specialist(latest);meta=state['metadata']
            if meta['dataset_hash']!=file_hash(data/'manifest.json'):raise ValueError('Resume dataset changed')
            completed=meta.get('completed',[]);phase_index=meta['phase_index'];start_epoch=meta['epoch']
            if meta.get('accepted'):status(state='complete',completed=completed);return
        else:model=load_legacy(args.initial_checkpoint,spec)
        model.to('cuda');torch.manual_seed(719)
        status(state='evaluating',phase='baseline',parameters=sum(p.numel() for p in model.parameters()),
            source=args.initial_checkpoint,dataset=manifest['counts'])
        fixed=balanced(dev,28,720);baseline=evaluate(model,fixed)
        if not (out/'baseline.json').exists():atomic(out/'baseline.json',baseline)
        # A disposable overfit test checks optimization before any long run.
        if not state and not args.skip_overfit:
            probe=copy.deepcopy(model);probe_rows=balanced([r for r in train if r['phase'] in ('multiply','power')],32,721)
            optim=torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad],lr=1e-4)
            before=evaluate(probe,probe_rows)
            for step in range(1,201):
                probe.train();optim.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=batch_loss(probe,probe_rows)
                loss.backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);optim.step()
                if step%50==0:
                    after=evaluate(probe,probe_rows)
                    status(state='overfit_test',step=step,steps=200,accuracy=after['accuracy'],trace_accuracy=after['trace_accuracy'])
                    if after['accuracy']>=.95:break
            atomic(out/'overfit_test.json',{'before':before,'after':after,'steps':step,'production_weights_changed':False})
            del probe,optim;torch.cuda.empty_cache()
            if after['accuracy']<.95:
                status(state='failed_acceptance',phase='overfit_test',reason='32-example optimization test failed; long training not started');return
        for index in range(phase_index,len(PHASES)):
            phase=PHASES[index]
            if phase in completed:continue
            active=[r for r in train if r['phase']==phase];panel=[r for r in dev if r['phase']==phase][:64]
            replay=[r for r in train if r['phase']!=phase]
            retention=balanced([r for r in dev if r['phase']!=phase],28,722)
            baseline_path=out/f'{phase}_before.json'
            if baseline_path.exists():entry=json.loads(baseline_path.read_text())
            else:
                entry={'active':evaluate(model,panel),'retention':evaluate(model,retention)};atomic(baseline_path,entry)
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5)
            consecutive=0
            if state and state.get('optimizer') is not None and index==phase_index:
                optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'])
                if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
                consecutive=state['metadata'].get('consecutive',0)
            for epoch in range(start_epoch+1 if index==phase_index else 1,args.epochs+1):
                begun=time.time();model.train();rng=random.Random(719+index*10000+epoch)
                # Each active row is seen once per epoch, length bucketing reduces padding.
                ordered=list(active);rng.shuffle(ordered)
                buckets=[sorted(ordered[i:i+192],key=lambda r:len(r['prompt'])+len(r['target'])) for i in range(0,len(ordered),192)]
                losses=[]
                for chunk in buckets:
                    for offset in range(0,len(chunk),24):
                        rows=chunk[offset:offset+24]+balanced(replay,8,719+epoch*10000+len(losses))
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.bfloat16):loss=batch_loss(model,rows)
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
                        if len(losses)%10==0:status(state='training',phase=phase,epoch=epoch,epochs=args.epochs,
                            step=len(losses),loss=losses[-1],active_examples=len(active),gpu_bytes=torch.cuda.memory_allocated())
                observed=evaluate(model,panel);retained=evaluate(model,retention)
                passed=observed['accuracy']>=.95 and observed['trace_accuracy']>=.90 and retained['accuracy']>=entry['retention']['accuracy']
                consecutive=consecutive+1 if passed else 0
                report={'phase':phase,'epoch':epoch,'loss':sum(losses)/len(losses),'active':observed,'retention':retained,
                    'retention_baseline':entry['retention']['accuracy'],'passed':passed,'consecutive':consecutive,'elapsed':time.time()-begun}
                atomic(out/f'{phase}_epoch_{epoch:03d}.json',report)
                metadata={'accepted':False,'dataset_hash':file_hash(data/'manifest.json'),'phase_index':index,'epoch':epoch,
                    'completed':completed,'consecutive':consecutive,'report':report,'source_checkpoint':str(args.initial_checkpoint)}
                save_specialist(latest,model,metadata,optimizer)
                status(state='evaluated',phase=phase,epoch=epoch,accuracy=observed['accuracy'],retention=retained['accuracy'],passed=passed)
                if (out/'STOP').exists():status(state='paused',phase=phase,epoch=epoch,checkpoint=str(latest));return
                if consecutive>=2:
                    sealed=evaluate(model,[r for r in test if r['phase']==phase][:64]);atomic(out/f'{phase}_test.json',sealed)
                    if sealed['accuracy']<.95:status(state='failed_acceptance',phase=phase,reason='sealed test failed');return
                    completed.append(phase);metadata.update(completed=completed,phase_index=index+1,epoch=0,consecutive=0)
                    save_specialist(latest,model,metadata)
                    break
            else:status(state='failed_acceptance',phase=phase,reason='bounded epoch budget reached');return
            state=None;start_epoch=0
        # Retest every task after all sequential updates, not just at acquisition time.
        final={p:evaluate(model,[r for r in test if r['phase']==p][:64]) for p in PHASES}
        accepted=all(v['accuracy']>=.95 for v in final.values())
        atomic(out/'final.json',{'accepted':accepted,'domains':final})
        save_specialist(out/'math.specialist',model,{'accepted':accepted,'completed':completed,'dataset_hash':file_hash(data/'manifest.json'),'final':final})
        status(state='complete' if accepted else 'failed_acceptance',accepted=accepted,checkpoint=str(out/'math.specialist'))
    except Exception as exc:
        status(state='failed',error=str(exc));raise
    finally:lock.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--initial-checkpoint',required=True);p.add_argument('--config',required=True)
    p.add_argument('--data',required=True);p.add_argument('--dataset-output',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--skip-overfit',action='store_true')
    run(p.parse_args())


if __name__=='__main__':main()
