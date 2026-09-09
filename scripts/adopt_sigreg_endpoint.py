"""Explicit one-time adoption of the user-selected paired-trial endpoint."""
import copy
import json
import shutil
from pathlib import Path
import torch
from cftn_v3.data import file_hash
from cftn_v3.file_io import atomic_write,atomic_json
from cftn_v3.criterion_repair import failures

def adopted_metadata(anchor,experiment,endpoint_report,policy):
    meta=copy.deepcopy(anchor);rounds=experiment['rounds']
    assert meta['dataset_hash']==experiment['dataset_sha256']
    assert meta['stage_index']==experiment['stage_index']
    assert meta['controller']['mode']=='normal'
    meta.update(round=experiment['seed_start_round']+rounds,cursor=0,policy=policy,accepted=False,consecutive=0)
    c=meta['controller'];c['normal_done']+=rounds;c['normal_total']+=rounds
    c.update(streak=0,full_streak=0,full_pass_round=None)
    meta['weak']=failures(endpoint_report['active'])
    meta['retention_weak']=failures(endpoint_report['retention'],True)
    meta['adopted_sigreg_experiment']={'checkpoint_sha256':experiment['checkpoint_sha256'],'rounds':rounds,'coefficient':experiment['coefficient']}
    return meta

def main():
    root=Path('G:/ctfn-text/artifacts/v3_curriculum_v4');trial=root/'sigreg_comparison';out=root/'math'
    assert not (root/'native_training.lock').exists(),'Worker lock exists'
    assert not (out/'sigreg_adoption.json').exists(),'Endpoint already adopted'
    spec=json.loads((trial/'experiment.json').read_text());cfg=json.loads(Path('config/local_curriculum_v4.json').read_text());policy=cfg['queue'][0]['policy']
    assert policy['sigreg_coefficient']==spec['coefficient']
    assert file_hash(trial/'start.specialist')==spec['checkpoint_sha256']
    anchor=torch.load(trial/'start.specialist',map_location='cpu',weights_only=True)
    endpoint=torch.load(trial/'sigreg/endpoint.specialist',map_location='cpu',weights_only=True)
    state=torch.load(trial/'sigreg/current.specialist',map_location='cpu',weights_only=True)
    assert state['metadata']['experiment_round']==11 and state['metadata']['cursor']==0
    assert endpoint['metadata']['dataset_hash']==spec['dataset_sha256'] and endpoint['optimizer']
    assert all(torch.equal(v,state['weights'][k]) for k,v in endpoint['weights'].items())
    report=json.loads((trial/'sigreg/endpoint.json').read_text())
    meta=adopted_metadata(anchor['metadata'],spec,report,policy);phase=json.loads((out/'curriculum.json').read_text())['stages'][spec['stage_index']]['name']
    meta['adopted_sigreg_experiment']['endpoint']=str(trial/'sigreg/endpoint.specialist')
    backup=out/'before_sigreg_adoption.specialist';assert not backup.exists();shutil.copy2(out/'current.specialist',backup)
    endpoint['metadata']=meta
    atomic_write(out/'current.specialist',lambda f:torch.save(endpoint,f))
    for i in range(1,spec['rounds']+1):
        r=json.loads((trial/'sigreg'/f'{phase}_epoch_{i:03d}.json').read_text());r['epoch']=spec['seed_start_round']+i-1
        destination=out/f"{phase}_epoch_{r['epoch']:03d}.json";assert not destination.exists()
        r['passed']=not failures(r['active']) and not failures(r['retention'],True)
        atomic_json(destination,r)
    old=out/f'{phase}_promotion_validation.json'
    if old.exists():shutil.copy2(old,trial/'pre_adoption_promotion.json')
    atomic_json(old,{'phase':phase,'epoch':meta['round']-1,'active':report['active'],'retention':report['retention'],
        'passed':not meta['weak'] and not meta['retention_weak'],'full_consecutive':0,'adopted_experiment':True})
    atomic_json(out/'sigreg_adoption.json',{'source':str(trial/'sigreg/endpoint.specialist'),'source_sha256':file_hash(trial/'sigreg/endpoint.specialist'),
        'backup':str(backup),'resume_round':meta['round'],'normal_done':meta['controller']['normal_done'],'policy':policy})
    print({'resume_round':meta['round'],'normal_completed':meta['controller']['normal_done'],'normal_budget':policy['normal_rounds'],'coefficient':policy['sigreg_coefficient']})

if __name__=='__main__':main()
