import json,shutil,torch
from pathlib import Path
from cftn_v3.full_curriculum_data import verify_manifest
from cftn_v3.data import file_hash
from cftn_v3.file_io import atomic_write
root=Path('G:/ctfn-text/artifacts/v3_1/math');data=Path('G:/ctfn-text/data/v3_1_foundation_v3')
assert json.loads((root/'status.json').read_text())['state']=='paused'
man=verify_manifest(data);p=root/'current.specialist';s=torch.load(p,map_location='cpu',weights_only=True);m=s['metadata']
assert m['stage_index']==0 and not m.get('terminal') and m['dataset_hash']==man['parent_manifest_sha256']
backup=root/f"before_foundation_expansion_round_{m['round']}.specialist";assert not backup.exists();shutil.copy2(p,backup)
m['dataset_hash']=file_hash(data/'manifest.json');m['cursor']=0;m['consecutive']=0
m['controller'].update(mode='repair',focus='1NPV-2',recovery_queue=['1NPV-2','1NPV-1'],repair_done=0,recovery_gate_streak=0,focus_streak=0,streak=0,full_streak=0,full_pass_round=None,validation_enabled=True,validation_start_round=m['round'],normal_done=0,recovery_blocks=0,attempt_counts={})
atomic_write(p,lambda stream:torch.save(s,stream))
config=Path('config/local_curriculum_v31.json');cfg=json.loads(config.read_text());cfg['queue'][0]['data']=str(data);config.write_text(json.dumps(cfg,indent=2)+'\n')
report={'backup':str(backup),'data':str(data),'round':m['round'],'validation':'100% active/trace/format, full focused checks every five rounds and next after pass'}
(root/'foundation_expansion_adoption.json').write_text(json.dumps(report));print(json.dumps(report))
