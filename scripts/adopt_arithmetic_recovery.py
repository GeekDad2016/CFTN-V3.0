"""Explicit, backed-up migration of the paused math checkpoint to recovery v5."""
import json,shutil
from pathlib import Path
import torch
from cftn_v3.data import file_hash
from cftn_v3.full_curriculum_data import verify_manifest
from cftn_v3.file_io import atomic_write

root=Path('G:/ctfn-text/artifacts/v3_curriculum_v4/math')
data=Path('G:/ctfn-text/data/v3_math_recovery_v5')
status=json.loads((root/'status.json').read_text());assert status['state']=='paused'
manifest=verify_manifest(data)
p=root/'current.specialist';saved=torch.load(p,map_location='cpu',weights_only=True);meta=saved['metadata']
assert meta['stage_index']==5 and not meta.get('terminal')
assert meta['dataset_hash']==manifest['parent_manifest_sha256']
backup=root/f"before_arithmetic_recovery_round_{meta['round']}.specialist"
assert not backup.exists();shutil.copy2(p,backup)
old_cursor=meta['cursor'];meta['cursor']=0
meta['dataset_hash']=file_hash(data/'manifest.json');meta['consecutive']=0
queue=['RECOVERY-PARTIAL-SUM','EXT-MULTI-DIGIT-SUBTRACT','KS2-EXACT-DIVIDE','KS2-LONG-MULTIPLY']
meta['controller'].update(mode='repair',focus=queue[0],recovery_queue=queue,repair_done=0,recovery_gate_streak=0,focus_streak=0,streak=0,full_streak=0,full_pass_round=None,validation_enabled=True,validation_start_round=meta['round'],normal_done=0,recovery_blocks=0,attempt_counts={})
meta['arithmetic_recovery']={'backup':str(backup),'data':str(data),'restarted_round_cursor':old_cursor}
atomic_write(p,lambda stream:torch.save(saved,stream))
config=Path('config/local_curriculum_v4.json');cfg=json.loads(config.read_text());cfg['queue'][0]['data']=str(data)
config.write_text(json.dumps(cfg,indent=2)+'\n')
(root/'arithmetic_recovery_adoption.json').write_text(json.dumps(meta['arithmetic_recovery']))
print(json.dumps({'round':meta['round'],'backup':str(backup),'queue':queue}))
