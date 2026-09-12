"""One-time validation-only migration; run with the training worker stopped."""
import json
import shutil
from pathlib import Path
import torch
from cftn_v3.data import file_hash
from cftn_v3.full_curriculum_data import verify_manifest
from cftn_v3.file_io import atomic_write, atomic_json

root = Path('G:/ctfn-text/artifacts/v3_1')
source = Path('G:/ctfn-text/data/v3_1_foundation_v3')
dest = Path('G:/ctfn-text/data/v3_1_balanced_validation_v4')
assert not (root/'native_training.lock').exists()
assert not (root/'pipeline.lock').exists()
m = verify_manifest(dest)
for split in ('train', 'test'):
    assert file_hash(source/f'{split}.jsonl') == file_hash(dest/f'{split}.jsonl')
old = [json.loads(s) for s in (source/'validation.jsonl').read_text().splitlines()]
new = [json.loads(s) for s in (dest/'validation.jsonl').read_text().splitlines()]
assert new[:len(old)] == old and len(new)-len(old) == 269
path = root/'math/current.specialist'
p = torch.load(path, map_location='cpu', weights_only=True)
meta = p['metadata']
assert meta['dataset_hash'] == m['parent_manifest_sha256'] == file_hash(source/'manifest.json')
assert meta['stage_index'] == 1 and not meta.get('terminal')
backup = path.with_name(f"before_balanced_validation_round_{meta['round']}.specialist")
assert not backup.exists()
shutil.copy2(path, backup)
meta['dataset_hash'] = file_hash(dest/'manifest.json')
meta['consecutive'] = 0
meta['controller'].update(streak=0, full_streak=0, full_pass_round=None)
atomic_write(path, lambda stream: torch.save(p, stream))
config_path = Path('config/local_curriculum_v31.json')
config = json.loads(config_path.read_text())
assert Path(config['queue'][0]['data']) == source
config['queue'][0]['data'] = str(dest)
config_path.write_text(json.dumps(config, indent=2)+'\n')
audit = {'backup': str(backup), 'round': meta['round'], 'cursor': meta['cursor'],
         'old_hash': m['parent_manifest_sha256'], 'new_hash': meta['dataset_hash'],
         'validation_records': len(new), 'stage2_validation_records': m['stages'][1]['validation_records'],
         'training_and_test_unchanged': True, 'pass_streaks_reset': True}
atomic_json(root/'math/balanced_validation_adoption.json', audit)
print(json.dumps(audit))
