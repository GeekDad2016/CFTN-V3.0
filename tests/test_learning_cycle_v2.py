import json
import pytest
import torch
from cftn_v3.learning_cycle_v2 import prepare_v2, select_pool, authorized
from cftn_v3.learning_experiment import record, DOMAINS
from cftn_v3.config import Config, canonical
from cftn_v3.model import CFTN, ByteTokenizer
from cftn_v3.training import train, make_plan
from cftn_v3.artifact import save_bundle, load_bundle
from cftn_v3.data import read_rows


def test_split_preserved_and_pool_rotates(tmp_path):
    old,new=tmp_path/'old',tmp_path/'new';old.mkdir()
    targets={'math':'<answer>2</answer>','code':'def solve(x):\n return x*2+3',
             'retrieval':'yes','formal_logic':'P(1);Q(1);R(1)'}
    for t in DOMAINS:
        for split in ('train','heldout'):
            (old/f'{t}_{split}.jsonl').write_text(canonical(record(t,t+split,targets[t],'test'))+'\n')
    (old/'manifest.json').write_text('{"sources":{}}')
    prepare_v2(old,new);prepare_v2(old,new)
    for t in DOMAINS:
        train_ids={r['semantic_id'] for r in read_rows(new/f'{t}_train.jsonl')}
        held_ids={r['semantic_id'] for r in read_rows(new/f'{t}_heldout.jsonl')}
        assert not train_ids & held_ids
        assert held_ids=={r['semantic_id'] for r in read_rows(old/f'{t}_heldout.jsonl')}
    pool=list(range(3000))
    assert len(set(select_pool(pool,0)))==1024
    assert not set(select_pool(pool,0)) & set(select_pool(pool,1))


def test_cuda_checkpoint_resume(tmp_path):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    config=Config.tiny();config.max_continual_steps=1000
    model=CFTN(config,ByteTokenizer()).to(device)
    row={**record('math','2+2?','<answer>4</answer>','test'),
         'contract_version':'canonical_v2','teacher_revision':Config().revision}
    plan=make_plan('continual',('math',),[row],verifier=authorized)
    state=train(model,[row],plan,1,verifier=authorized)
    save_bundle(tmp_path/'resume.cftn',model,training=state,metadata={'step':1})
    restored,saved,_=load_bundle(tmp_path/'resume.cftn',device,training=True)
    resumed=train(restored,[row],plan,1,state=saved,verifier=authorized)
    assert resumed['step']==2 and resumed['frozen_hashes_verified']
    assert all(n.startswith('towers.math.') for n in resumed['changed'])
    bad={**row,'target':'<answer>5</answer>'}
    assert not authorized(bad)
