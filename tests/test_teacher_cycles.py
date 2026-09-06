import pytest
from cftn_v3 import teacher_cycles as tc
from cftn_v3.live import Store
from cftn_v3.config import Config
from cftn_v3.model import CFTN, ByteTokenizer
from cftn_v3.training import make_plan, train
from cftn_v3.data import example


@pytest.fixture
def row(monkeypatch):
    r = tc.reference_row({'question': 'Two plus three?', 'answer': '2+3=5\n#### 5'}, 'train', 0)
    monkeypatch.setattr(tc, 'catalog', lambda root: {r['id']: r})
    return r


def test_reference_verification_rejects_wrong_and_unchecked_work(row):
    assert tc.verify(row, '<answer>5.0</answer>')
    assert not tc.verify(row, '<answer>6</answer>')
    assert not tc.verify(row, 'Wrong reasoning <answer>5</answer>')
    assert not tc.verify({**row, 'prompt': 'Different question'}, row['target'])
    assert not tc.verify({**row, 'tower': 'code'}, row['target'])


def test_offline_import_public_boundary_and_holdout(row, tmp_path):
    store = Store(tmp_path/'live.sqlite')
    assert not store.ingest(row)['verified']
    tc.import_verified(store, [row])
    with pytest.raises(ValueError):
        tc.import_verified(store, [{**row, 'source_split': 'test'}])
    store.db.close()


def test_teacher_row_selective_cuda_update(row):
    import torch
    model = CFTN(Config.tiny(), ByteTokenizer()).to('cuda' if torch.cuda.is_available() else 'cpu')
    state = train(model, [row], make_plan('continual', ('math',), [row]), 1,
                  replay=[example('math', 0, 'en')])
    assert state['frozen_hashes_verified']
    assert state['changed'] and all(k.startswith('towers.math.') for k in state['changed'])
