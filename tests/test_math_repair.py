from cftn_v3.math_repair import curriculum,make_row,verify,panel

def test_same_range_disjoint_and_balanced():
    rows=curriculum()
    ids=[{r['semantic_id'] for r in rows[s]} for s in ('train','development','test')]
    assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
    for split in ('train','development','test'):
        assert all(0<=r['a']<=r['b']<100 for r in rows[split])
        assert len(panel(rows[split]))==64
        assert sum(r['criterion']=='carry' for r in panel(rows[split]))==32
    r=make_row(12,29)
    assert verify(r)
    assert not verify(r,'<answer>40</answer>')
    assert not verify({**r,'prompt':'Different prompt'})

def test_repair_cuda_update_isolated():
    import torch
    from cftn_v3.config import Config
    from cftn_v3.model import CFTN,ByteTokenizer
    from cftn_v3.training import train,make_plan
    from cftn_v3.data import example
    rows=[make_row(12,29)]
    model=CFTN(Config.tiny(),ByteTokenizer()).to('cuda' if torch.cuda.is_available() else 'cpu')
    state=train(model,rows,make_plan('specialist',('math',),rows),1,replay=[example('math',0,'en')])
    assert state['frozen_hashes_verified']
    assert all(k.startswith('towers.math.') for k in state['changed'])
