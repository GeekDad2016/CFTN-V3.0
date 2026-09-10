"""Switch the paused arithmetic recovery checkpoint to merged normal training."""
import json,shutil
from pathlib import Path
import torch
from cftn_v3.file_io import atomic_write
from cftn_v3.full_curriculum_data import verify_manifest
from cftn_v3.data import file_hash
root=Path('G:/ctfn-text/artifacts/v3_curriculum_v4/math')
assert json.loads((root/'status.json').read_text())['state']=='paused'
p=root/'current.specialist';saved=torch.load(p,map_location='cpu',weights_only=True);m=saved['metadata']
config=Path('config/local_curriculum_v4.json');cfg=json.loads(config.read_text());item=cfg['queue'][0]
verify_manifest(item['data']);assert m['dataset_hash']==file_hash(Path(item['data'])/'manifest.json')
assert m['stage_index']==5 and not m.get('terminal')
backup=root/f"before_merged_normal_round_{m['round']}.specialist"
assert not backup.exists();shutil.copy2(p,backup)
m['policy']['validation_loss_threshold']=.003
m['controller'].update(mode='normal',focus=None,recovery_queue=[],recovery_gate_streak=0,repair_done=0,focus_streak=0,normal_done=0,streak=0,full_streak=0,full_pass_round=None,validation_enabled=False,recovery_blocks=0,attempt_counts={})
m['consecutive']=0;m['cursor']=0
m['merged_normal_adoption']={'backup':str(backup),'threshold':.003}
atomic_write(p,lambda stream:torch.save(saved,stream))
item['policy']['validation_loss_threshold']=.003
config.write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps({'round':m['round'],'backup':str(backup),'threshold':.003}))
