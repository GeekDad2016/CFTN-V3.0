import json
import zipfile
from pathlib import Path
import pytest
import torch

from cftn_v3.config import Config, TOWERS
from cftn_v3.model import CFTN, ByteTokenizer
from cftn_v3.contracts import Call, ExecutionPlan, UpdatePlan, Evidence
from cftn_v3.data import example, verify, sample_update, prepare, audit
from cftn_v3.training import make_plan, train, parameter_hashes
from cftn_v3.artifact import save_bundle, load_bundle
from cftn_v3.live import Store, GPULock
from cftn_v3.evaluation import gate, release_gate


@pytest.fixture
def model():
    torch.manual_seed(719)
    return CFTN(Config.tiny())


def test_bilingual_exact_bytes():
    text = 'English / Română: ăâîșț ĂÂÎȘȚ'
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text


@pytest.mark.parametrize('tower', TOWERS)
@pytest.mark.parametrize('language', ('en', 'ro'))
def test_twelve_verified_capabilities(tower, language):
    row = example(tower, 93, language)
    assert verify(row)
    assert not verify(row, 'wrong result')


def test_execution_and_update_are_separate():
    ExecutionPlan([Call('math', 0, 'x'), Call('string', 1, 'y', ('math',))]).validate(TOWERS)
    with pytest.raises(ValueError):
        ExecutionPlan([Call('math', 0, 'x', ('string',))]).validate(TOWERS)
    with pytest.raises(ValueError):
        ExecutionPlan([Call('math', 0, 'x')]).validate(())
    with pytest.raises(ValueError):
        UpdatePlan('specialist', ('math',), ()).validate()


def test_only_target_parameters_change(model):
    row = example('math', 4, 'ro')
    before = parameter_hashes(model)
    plan = make_plan('specialist', ('math',), [row])
    state = train(model, [row], plan, 1)
    after = parameter_hashes(model)
    changed = [k for k in before if before[k] != after[k]]
    assert changed and all(k.startswith('towers.math.') for k in changed)
    assert state['frozen_hashes_verified']
    assert all(not p.requires_grad for p in model.dispatcher.parameters())


def test_stable_routing_features_ignore_adapter(model):
    ids = model.ids('Calculează 2+2')
    first = model.coordinator.stable_features(ids)
    with torch.no_grad():
        model.coordinator.adapter[-1].weight.fill_(1)
    assert torch.equal(first, model.coordinator.stable_features(ids))


def test_artifact_roundtrip_and_resume(model, tmp_path):
    row = example('math', 3, 'en')
    plan = make_plan('specialist', ('math',), [row])
    state = train(model, [row], plan, 1)
    path = tmp_path/'model.cftn'
    save_bundle(path, model, training=state)
    loaded, restored, _ = load_bundle(path, training=True)
    assert parameter_hashes(loaded) == parameter_hashes(model)
    assert restored['step'] == 1
    train(model, [row], plan, 1, state=state)
    train(loaded, [row], plan, 1, state=restored)
    assert parameter_hashes(loaded) == parameter_hashes(model)


def test_tampered_bundle_rejected(model, tmp_path):
    path = tmp_path/'m.cftn'
    save_bundle(path, model)
    with zipfile.ZipFile(path, 'a') as z:
        z.writestr('../bad', 'bad')
    with pytest.raises(ValueError): load_bundle(path)


def test_replay_balanced_languages():
    new = [example('math', i, 'ro') for i in range(8)]
    old = [example('math', 100+i, lang) for i in range(4) for lang in ('en', 'ro')]
    rows = sample_update(new, old, 32, 719)
    assert sum(r['index'] >= 100 for r in rows) == 8
    assert sum(r['index'] >= 100 and r['language'] == 'en' for r in rows) == 4


def test_data_audit(tmp_path):
    assert prepare(tmp_path/'data', train_objects=3, panel_objects=2)['status'] == 'passed'
    path = tmp_path/'data'/'train.jsonl'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError): audit(tmp_path/'data')


def test_private_confirmation_and_idempotence(tmp_path):
    store = Store(tmp_path/'live.sqlite')
    row = {'tower': 'math', 'language': 'ro', 'prompt': 'nou', 'target': '4',
           'verifier': 'human_confirmed_v1', 'confirmation_id': 'spoof'}
    assert not store.ingest(row)['verified']
    assert store.ingest(row, confirm=True)['verified']
    store.ingest(row, confirm=True)
    assert store.db.execute('SELECT COUNT(*) FROM interactions').fetchone()[0] == 1
    assert not store.pending('math', force=True)


def test_missing_panels_fail_closed():
    assert not gate({'groups': {}})['passed']
    assert not release_gate([], [], {}, {})['passed']


def test_gpu_lock(tmp_path):
    with GPULock(tmp_path):
        with pytest.raises(RuntimeError):
            with GPULock(tmp_path): pass
    assert not (tmp_path/'gpu.lock').exists()


def test_gradient_through_frozen_bridge(model):
    bridge = model.bridges['math']['request']
    bridge.requires_grad_(False)
    source = torch.randn(1, 3, model.coordinator.width, requires_grad=True)
    bridge(source).sum().backward()
    assert source.grad.abs().sum() > 0
    assert all(p.grad is None for p in bridge.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_selective_update_and_export(tmp_path):
    model = CFTN(Config.tiny()).cuda()
    row = example('math', 7, 'ro')
    result = train(model, [row], make_plan('specialist', ('math',), [row]), 1)
    assert result['frozen_hashes_verified'] and result['loss'] > 0
    save_bundle(tmp_path/'cuda.cftn', model)
    loaded, _, _ = load_bundle(tmp_path/'cuda.cftn', 'cuda')
    assert parameter_hashes(model) == parameter_hashes(loaded)
