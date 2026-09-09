"""Persistent sequential handoff: a stopped or failed worker is never acceptance."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from .local_math_training import atomic
from .data import file_hash

def run(config):
    config=Path(config);cfg=json.loads(config.read_text());root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    lock=root/'pipeline.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        for item in cfg['queue']:
            out=Path(item['output']);out.mkdir(parents=True,exist_ok=True)
            atomic(root/'queue.json',{'state':'running','current':item['tower'],'output':str(out),'pid':os.getpid(),'queue':cfg['queue'],'updated':time.time()})
            trainer=item.get('trainer','standard')
            if trainer not in ('standard','criterion'):raise ValueError('Unknown trainer')
            module='cftn_v3.criterion_curriculum_training' if trainer=='criterion' else 'cftn_v3.full_curriculum_training'
            command=[sys.executable,'-u','-m',module,'--data',item['data'],'--output',str(out),'--initial-checkpoint',item['initial_checkpoint']]
            if item.get('inherit_progress'):command.append('--inherit-progress')
            if item.get('upgrade_validation_warmup'):command.append('--upgrade-validation-warmup')
            if item.get('stage_first'):command.append('--stage-first')
            if item.get('upgrade_recovery'):command.append('--upgrade-recovery')
            for key,value in item.get('policy',{}).items():
                if key not in ('normal_rounds','remediation_rounds','attempts','examples','lr','validation_examples','retention_examples','consolidation_rounds','stage_rounds','full_check_every','sigreg_coefficient','validation_warmup_rounds','validation_loss_threshold'):
                    raise ValueError('Unknown training policy setting: '+key)
                command.extend(['--'+key.replace('_','-'),str(value)])
            with (out/'training.stdout.log').open('a') as stdout,(out/'training.stderr.log').open('a') as stderr:
                result=subprocess.run(command,stdout=stdout,stderr=stderr)
            status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
            artifact=out/(item['tower']+'.specialist')
            if result.returncode or status.get('state')!='complete' or not status.get('accepted') or not artifact.exists():
                atomic(root/'queue.json',{'state':'blocked','current':item['tower'],'output':str(out),'reason':status.get('error') or status.get('reason') or 'Worker did not pass acceptance','queue':cfg['queue']});return
            # Verify the actual native artifact and dataset binding before handing over the GPU.
            import torch
            saved=torch.load(artifact,map_location='cpu',weights_only=True)
            if not saved['metadata'].get('accepted') or saved['metadata']['dataset_hash']!=file_hash(Path(item['data'])/'manifest.json'):
                raise ValueError('Accepted artifact does not match queued dataset')
            if (out.parent/'native_training.lock').exists():raise RuntimeError('GPU worker lock remains after exit')
            del saved
        atomic(root/'queue.json',{'state':'complete','reason':'Every queued specialist passed','queue':cfg['queue']})
    except Exception as exc:
        atomic(root/'queue.json',{'state':'failed','error':str(exc),'queue':cfg['queue']});raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
