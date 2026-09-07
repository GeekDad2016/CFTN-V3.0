"""Bounded disposable CUDA update and checkpoint/resume check; does not train the live model."""
import json
from pathlib import Path
import torch
from cftn_v3.local_specialist import load_specialist,save_specialist
from cftn_v3.local_math_training import batch_loss,read
from cftn_v3.criterion_repair import RepairController
from cftn_v3.balanced_curriculum_data import comparison_family

def main():
    torch.set_num_threads(4)
    data=Path('G:/ctfn-text/data/v3_full_math_v4')
    rows=read(data/'train.jsonl');old=read(Path('G:/ctfn-text/data/v3_full_math_v3/train.jsonl'))
    old_ids={r['semantic_id'] for r in old};added=[r for r in rows if r['semantic_id'] not in old_ids]
    held=read(data/'validation.jsonl')+read(data/'test.jsonl')
    held += [r['record'] for r in read(data/'quarantined_heldout.jsonl')]
    assert not {comparison_family(r['ir']) for r in added}&{comparison_family(r['ir']) for r in held}
    model,saved=load_specialist('G:/ctfn-text/artifacts/v3_curriculum_v3/math/current.specialist')
    assert saved['metadata']['stage_index']==1 and saved['metadata']['completed']==['y1_number_structure']
    model.to('cuda');optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    optimizer.load_state_dict(saved['optimizer']);controller=RepairController();controller.focus(['EXT-COMPARE-EXPRESSIONS'],[])
    batch=controller.rows([r for r in rows if r['stage']==1],[r for r in rows if r['stage']==0],12,9307)
    assert {r['criterion'] for r in batch}=={'EXT-COMPARE-EXPRESSIONS'}
    model.train();optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=batch_loss(model,batch)
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    output=Path('G:/ctfn-text/artifacts/v3_curriculum_v4_validation');output.mkdir(exist_ok=True)
    save_specialist(output/'smoke.specialist',model,{'tower':'math','controller':controller.state,'cursor':1},optimizer)
    restored,check=load_specialist(output/'smoke.specialist')
    assert check['metadata']['controller']==controller.state and check['metadata']['cursor']==1
    assert all(torch.equal(p.cpu(),restored.state_dict()[k]) for k,p in model.state_dict().items())
    result={'cuda':torch.cuda.get_device_name(),'finite_loss':float(loss.detach()),'new_rows_no_heldout_family_overlap':len(added),'checkpoint_roundtrip':True,'source_stage_index':saved['metadata']['stage_index']}
    (output/'result.json').write_text(json.dumps(result),encoding='utf-8');print(result)

if __name__=='__main__':main()
