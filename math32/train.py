"""Resumable standalone curriculum with sparse validation and exact retention gates."""
import argparse, collections, copy, json, math, os, random, time
from pathlib import Path
import torch
from torch.nn import functional as F
from cftn_v3.file_io import atomic_json,atomic_write,StatusPublisher
from cftn_v3.sigreg import sigreg
from .model import Tokenizer,Tower,batch
from .data import ROOT,answer,identity

def sample(rows,count,seed):
    rng=random.Random(seed);groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:groups[r['stage']][r['case']].append(r)
    stages=sorted(groups);result=[]
    if not stages:return result
    for i in range(count):
        stage=stages[i%len(stages)];cases=sorted(groups[stage]);case=cases[(i//len(stages))%len(cases)]
        result.append(rng.choice(groups[stage][case]))
    rng.shuffle(result);return result

def evaluate(model,rows,progress):
    samples=[]
    for i,r in enumerate(rows):
        text,complete=model.generate(r['ir'],max_tokens=max(32,len(model.tok.encode(r['answer']))+8))
        correct=complete and text==r['answer']
        samples.append({'criterion':r['criterion'],'phase':str(r['stage']),'prompt':json.dumps(r['ir']),
            'expected':r['answer'],'expected_trace':r['answer'],'output':text,'complete':complete,
            'answer_correct':correct,'trace_correct':correct,'format_correct':complete and '<' not in text})
        if i%16==0 or i==len(rows)-1:progress(i+1,len(rows))
    def metric(xs):return {'examples':len(xs),**{k:sum(x[v] for x in xs)/len(xs) if xs else 1.
        for k,v in [('accuracy','answer_correct'),('trace_accuracy','trace_correct'),('format_accuracy','format_correct')]}}
    result=metric(samples);result['samples']=samples;result['criteria']={}
    for name in sorted({r['criterion'] for r in rows}):
        result['criteria'][name]=metric([s for s in samples if s['criterion']==name])
        result['criteria'][name]['strata']={c:metric([s for r,s in zip(rows,samples) if r['criterion']==name and r['case']==c]) for c in sorted({r['case'] for r in rows if r['criterion']==name})}
    return result

def run(config):
    cfg=json.loads(Path(config).read_text());root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    all_dataset=cfg.get('training_scope')=='all_dataset'
    lock=root/'native_training.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'output':str(root)}).encode());os.close(fd)
    publisher=StatusPublisher(root/'status.json');display={}
    def status(**kw):display.update(kw);publisher({'pid':os.getpid(),'updated':time.time(),**display})
    try:
        torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
        data=Path(cfg['data']);manifest=json.loads((data/'manifest.json').read_text())
        import hashlib
        for f,digest in manifest['files'].items():assert hashlib.sha256((data/f).read_bytes()).hexdigest()==digest
        records={s:[json.loads(l) for l in (data/(s+'.jsonl')).read_text().splitlines()] for s in ('train','validation','test')}
        tok=Tokenizer(json.loads((data/'vocab.json').read_text()));model=Tower(tok,**cfg['model']).to('cuda')
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01)
        state={'stage':0,'round':1,'cursor':0,'updates':0,'completed':[],
               'validation_enabled':all_dataset,'normal_done':0,'streak':0,'mode':'normal','repair_done':0,'repair_cycles':0}
        path=root/'current.specialist';binding=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()
        if path.exists():
            p=torch.load(path,map_location='cpu',weights_only=True);assert p['dataset_hash']==binding
            # Duration and validation cadence can change after a clean pause;
            # architecture, data and optimizer scale cannot.
            prior_cfg=p['config']
            immutable=('data','seed','model','lr','lr_warmup_updates','sigreg','batch_size','training_scope')
            assert all(prior_cfg.get(k)==cfg.get(k) for k in immutable)
            model.load_state_dict(p['weights']);opt.load_state_dict(p['optimizer']);state=p['state']
            torch.set_rng_state(p['rng']);torch.cuda.set_rng_state_all(p['cuda_rng'])
        if all_dataset:
            gate=cfg.get('all_dataset_validation_loss',cfg['loss_threshold'])
            if state.get('all_dataset_validation_loss')!=gate:
                state['validation_enabled']=False
                state['all_dataset_validation_loss']=gate
        def save():
            payload={'format':'math32','config':cfg,'dataset_hash':binding,'vocab':tok.vocab,
                'weights':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),
                'state':copy.deepcopy(state),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
            atomic_write(path,lambda f:torch.save(payload,f))
        atomic_json(root/'curriculum.json',manifest)
        status(state='starting',parameters=sum(p.numel() for p in model.parameters()),dataset=str(data),policy=cfg,
               strict_gate=True,checkpoint=str(path),gpu=torch.cuda.get_device_name(),stage_count=len(manifest['stages']))
        if state.get('terminal'):status(state='blocked',reason=state['terminal']);return
        if not path.exists():save()
        while ((cfg.get('all_dataset_epochs') is None or state['round']<=cfg['all_dataset_epochs']) if all_dataset else state['stage']<len(manifest['stages'])):
            idx=state['stage'];stage=manifest['stages'][idx];phase=stage['name']
            active=records['train'] if all_dataset else [r for r in records['train'] if r['stage']==idx]
            prior=[] if all_dataset else [r for r in records['train'] if r['stage']<idx]
            validation=records['validation'] if all_dataset else [r for r in records['validation'] if r['stage']==idx]
            # Fixed cumulative retention panel, balanced across all accepted stages/cases.
            earlier=[] if all_dataset else [r for r in records['validation'] if r['stage']<idx]
            retention=list({r['id']:r for r in sample(earlier,min(len(earlier),512),cfg['seed'])}.values())
            def ev(rows,label):
                status(state='evaluating',evaluation=label,evaluation_done=0,evaluation_total=len(rows))
                return evaluate(model,rows,lambda n,t:status(evaluation_done=n,evaluation_total=t))
            round_=state['round'];seed=cfg['seed']+idx*100000+round_
            status(phase='all_criteria_full_dataset' if all_dataset else phase,
                scope='Every record in the V3.2 training split, shuffled once per epoch' if all_dataset else stage['scope'],
                stage_index=None if all_dataset else idx,epoch=round_,epochs=cfg.get('all_dataset_epochs') if all_dataset else cfg['normal_rounds'],
                completed=state['completed'],normal_done=state['normal_done'],training_mode=state['mode'],
                validation_suppressed=not state['validation_enabled'],validation_loss_threshold=cfg.get('all_dataset_validation_loss',cfg['loss_threshold']) if all_dataset else cfg['loss_threshold'],
                next_full_check=(round_ if state['streak'] else ((round_//cfg['eval_every'])+1)*cfg['eval_every']) if state['validation_enabled'] else None,
                active_examples=len(active),sigreg_coefficient=cfg['sigreg'],evaluation=None)
            n=cfg['examples']*3//4 if prior else cfg['examples']
            trainrows=list(active) if all_dataset else sample(active,n,seed)+sample(prior,cfg['examples']-n,seed+1)
            if all_dataset:random.Random(seed).shuffle(trainrows)
            if state['mode']=='repair':
                failed=set(state.get('repair_cases',[]));focused=[r for r in active if r['case'] in failed]
                if focused:trainrows=sample(focused,cfg['examples']*3//5,seed)+sample(active,cfg['examples']//5,seed+1)+sample(prior or active,cfg['examples']-cfg['examples']*4//5,seed+2)
            random.Random(seed).shuffle(trainrows);chunks=[trainrows[i:i+cfg['batch_size']] for i in range(0,len(trainrows),cfg['batch_size'])]
            if not state['cursor']:state.update(loss_sum=0.,loss_count=0)
            status(state='training',step=state['cursor'],steps=len(chunks))
            for step,chunk in enumerate(chunks):
                if step<state['cursor']:continue
                model.train();offsets=[random.Random(seed+step*1000+i).randrange(1,65) for i in range(len(chunk))]
                ids,roles,positions,ops,labels,ends=batch(tok,chunk,'cuda',offsets)
                opt.zero_grad(set_to_none=True)
                lr=cfg['lr']*min(1.,(state['updates']+1)/cfg['lr_warmup_updates'])
                for group in opt.param_groups:group['lr']=lr
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,h=model(ids,roles,positions,ops,True)
                    ce=F.cross_entropy(logits[:,:-1].reshape(-1,512),labels[:,1:].reshape(-1),ignore_index=-100)
                    penalty=sigreg(h[torch.arange(len(chunk),device='cuda'),torch.tensor(ends,device='cuda')],seed+step,projections=64)
                    loss=ce+cfg['sigreg']*penalty
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                state['updates']+=1;state['cursor']=step+1;state['loss_sum']+=float(ce.detach());state['loss_count']+=1
                if step%8==0:status(step=step+1,loss=float(ce.detach()),sigreg_loss=float(penalty.detach()),lr=lr)
                if state['updates']%100==0 or (root/'STOP').exists():save()
                if (root/'STOP').exists():status(state='paused',reason='Checkpoint, optimizer and cursor saved');return
            mean=state['loss_sum']/state['loss_count'];state['cursor']=0
            state['round']+=1
            gate=cfg.get('all_dataset_validation_loss',cfg['loss_threshold']) if all_dataset else cfg['loss_threshold']
            if not state['validation_enabled'] and mean<=gate:state['validation_enabled']=True
            if state['validation_enabled']:state['normal_done']+=1
            status(round_mean_loss=mean,normal_done=state['normal_done'])
            atomic_json(root/f'{phase}_training_only_{round_:06d}.json',{'epoch':round_,'loss':mean,'validation_enabled':state['validation_enabled']})
            save()
            manual=(root/'VALIDATE_REQUEST.json').exists()
            due=state['validation_enabled'] and (round_%cfg['eval_every']==0 if all_dataset else (round_%cfg['eval_every']==0 or state['streak'] or state['normal_done']>=cfg['normal_rounds'] or state['mode']=='repair' and state['repair_done']>=cfg['repair_rounds']))
            if manual or due:
                if manual:status(manual_validation='running')
                report={'phase':phase,'epoch':round_,'loss':mean,'updated':time.time(),'active':ev(validation,'Full stage validation'),
                        'retention':ev(retention,'Fixed cumulative retention')}
                passed=all(report[k][m]==1. for k in ('active','retention') for m in ('accuracy','trace_accuracy','format_accuracy'))
                report['passed']=passed
                if manual:
                    report['evaluation_scope']='full'
                    atomic_json(root/f'{phase}_manual_round_{round_:06d}.json',report)
                    atomic_json(root/f'{phase}_manual_validation.json',report)
                    (root/'VALIDATE_REQUEST.json').unlink(missing_ok=True)
                    status(manual_validation='complete',manual_validation_round=round_,evaluation=None)
                if all_dataset:
                    report.update(full_consecutive=0,promoted=False,evaluation_scope='full_dataset_validation')
                    atomic_json(root/f'all_criteria_epoch_{round_:03d}.json',report)
                    save()
                    continue
                if due:
                    state['streak']=state['streak']+1 if passed else 0
                    if state['mode']=='repair' and passed:
                        state.update(mode='normal',normal_done=0,streak=0)
                    report.update(full_consecutive=state['streak'],promoted=False)
                    atomic_json(root/f'{phase}_epoch_{round_:03d}.json',report)
                    if state['streak']>=2 and state['normal_done']>=3:
                        # A sealed test is used once, only after validation has established mastery.
                        sealed=ev([r for r in records['test'] if r['stage']==idx],'Sealed stage test')
                        atomic_json(root/f'{phase}_test.json',sealed)
                        if sealed['accuracy']<1. or sealed['trace_accuracy']<1.:
                            state['terminal']='Sealed test failed; stage paused for a new evaluation design';save();status(state='blocked',reason=state['terminal']);return
                        report['promoted']=True;state['completed'].append(phase)
                        state.update(stage=idx+1,round=1,cursor=0,normal_done=0,validation_enabled=False,streak=0,mode='normal',repair_done=0,repair_cycles=0)
                        save();stagepath=root/f'{phase}.specialist';atomic_write(stagepath,lambda f:f.write(path.read_bytes()))
                    elif state['normal_done']>=cfg['normal_rounds']:
                        if state['repair_cycles']>=4:
                            state['terminal']='Four repair cycles exhausted; inspect held-out errors';save();status(state='blocked',reason=state['terminal']);return
                        state.update(mode='repair',repair_done=0,normal_done=0,repair_cycles=state['repair_cycles']+1,
                            repair_cases=[c for metric in report['active']['criteria'].values() for c,m in metric['strata'].items() if m['accuracy']<1.])
                    atomic_json(root/f'{phase}_promotion_validation.json',report)
            if state['mode']=='repair':
                state['repair_done']+=1
                if state['repair_done']>=cfg['repair_rounds']:state.update(mode='normal',normal_done=0,streak=0)
            save()
        status(state='complete',accepted=True,reason='Full-dataset experiment completed' if all_dataset else 'All maths stages passed')
    except Exception as e:status(state='failed',reason=str(e));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/math32.json');run(p.parse_args().config)
