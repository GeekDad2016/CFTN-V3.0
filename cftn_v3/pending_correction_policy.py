"""Apply an authorized post-experiment policy before any new GPU worker starts."""
import json
import shutil
from pathlib import Path
import torch
from .file_io import atomic_json, atomic_write

def apply(config,cfg,root):
    pending=root/'pending_correction_policy.json'
    if not pending.exists():return cfg
    if (root/'native_training.lock').exists():raise RuntimeError('Cannot apply policy while GPU worker is active')
    request=json.loads(pending.read_text())
    decision_path=root/'generated_correction_probe/decision.json'
    if not decision_path.exists():raise RuntimeError('Correction trial has not finished; pending policy retained')
    decision=json.loads(decision_path.read_text())
    enable=int(decision['selected']=='targeted')
    item=next(x for x in cfg['queue'] if x['tower']=='math')
    path=Path(item['output'])/'current.specialist'
    payload=torch.load(path,map_location='cpu',weights_only=True);meta=payload['metadata']
    backup=path.with_name(f"before_stage_corrections_round_{meta['round']}.specialist")
    if not backup.exists():shutil.copy2(path,backup)
    updates=dict(sigreg_coefficient=request['sigreg_coefficient'],validation_loss_threshold=request['validation_loss_threshold'],
                 unbounded_loss_warmup=1,validation_warmup_rounds=0,generated_correction=enable)
    item['policy'].update(updates);meta['policy'].update(updates)
    meta['controller'].update(mode='normal',focus=None,recovery_queue=[],training_only_rounds_remaining=0,
        validation_enabled=False,streak=0,full_streak=0,full_pass_round=None,correction_ids=[],correction_rounds_remaining=0)
    meta['consecutive']=0
    atomic_write(path,lambda f:torch.save(payload,f));atomic_json(config,cfg)
    atomic_json(root/'correction_policy_applied.json',{'enabled':bool(enable),'trial_selected':decision['selected'],
                'round':meta['round'],'backup':str(backup),'policy':updates})
    pending.unlink();return cfg
