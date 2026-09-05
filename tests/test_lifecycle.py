import json
import pytest
import torch

from cftn_v3.config import TOWERS, Config
from cftn_v3.data import example, verify, reservoir
from cftn_v3.model import CFTN
from cftn_v3.live import Store
from cftn_v3.artifact import save_bundle
from cftn_v3.evaluation import gate, release_gate
from cftn_v3.contracts import ExecutionPlan, Call


def test_code_is_functionally_checked_and_no_execution():
    row = example('code', 7, 'ro')
    assert verify(row, 'def solve(x):\n    return 9 + x')
    assert not verify(row, "import os\ndef solve(x):\n    return x + 9")
    assert not verify(row, "def solve(x):\n    return __import__('os').system('x')")


def test_sql_is_read_only_and_result_checked():
    row = example('structured_data', 7, 'en')
    assert verify(row, 'select value from items where 7=id')
    assert not verify(row, 'DELETE FROM items')
    assert not verify(row, "ATTACH DATABASE '/tmp/a' AS a")
    assert not verify(row, 'select value from items')


def test_feedback_and_fact_supersession(tmp_path):
    store = Store(tmp_path/'state.sqlite')
    event = store.event('Care este codul?', 'vechi', 'ro', 'r1')
    first = store.feedback(event, 'retrieval', 'nou', knowledge_kind='fact', stable_fact=False)
    assert not first['verified']
    second = store.feedback(event, 'retrieval', 'stabil', knowledge_kind='fact', stable_fact=True, supersedes=first['id'])
    assert second['verified']
    assert 'stabil' in store.memory('codul')[0]
    assert store.db.execute('SELECT superseded FROM interactions WHERE id=?', (first['id'],)).fetchone()[0] == 1


def test_queue_thresholds_and_consumption(tmp_path):
    store = Store(tmp_path/'state.sqlite')
    for i in range(32): store.ingest(example('math', i, 'ro'))
    assert not store.pending('math')
    assert len(store.pending('math', force=True)) == 32
    model = CFTN(Config.tiny())
    path = tmp_path/'accepted.cftn'
    save_bundle(path, model)
    consumed = [r['id'] for r in store.pending('math', force=True)]
    with pytest.raises(ValueError): store.activate(path, {'passed': False})
    store.activate(path, {'passed': True}, consumed)
    assert not store.pending('math', force=True)


def test_romanian_cannot_hide_behind_english():
    report = {'groups': {'math:en': {'count': 500, 'accuracy': 1.},
                         'math:ro': {'count': 500, 'accuracy': .5}}}
    assert not gate(report, targets=('math',))['passed']


def test_release_requires_two_seeds_and_integration():
    report = {'seed': 719, 'groups': {f'{t}:{l}': {'count': 500, 'accuracy': 1.} for t in TOWERS for l in ('en','ro')}}
    assert not release_gate([report,report], [], {}, {})['passed']
    second = {**report, 'seed': 1729}
    assert release_gate([report,second], [], {'accuracy_drop':0, 'validity':1}, {'accuracy':1,'inactive_calls':0})['passed']


def test_execution_capacity_limit():
    with pytest.raises(ValueError):
        ExecutionPlan([Call(t, 0, 'request') for t in TOWERS[:3]]).validate(TOWERS)


def test_reservoir_deterministic_and_bilingual():
    rows = [example('math', i, l) for i in range(20) for l in ('en','ro')]
    first = reservoir(rows, 12)
    assert first == reservoir(list(reversed(rows)), 12)
    assert sum(r['language']=='ro' for r in first) == 6


def test_replay_rotates_small_batch_across_criteria():
    from cftn_v3.data import sample_update
    rows = [example(t, 2, 'en') for t in TOWERS]
    observed = {sample_update(rows, [], 1, seed)[0]['tower'] for seed in range(100)}
    assert observed == set(TOWERS)


def test_composition_preserves_bilingual_split_identity():
    from cftn_v3.data import composition
    en, ro = composition(12, 'en'), composition(12, 'ro')
    assert en['semantic_id'] == ro['semantic_id']
    assert en['target'] == ro['target'] and verify(en) and verify(ro)
    assert ro['routing']['dependencies']['string'] == ['math']


def test_unrelated_optimizer_state_stays_identical():
    from cftn_v3.training import train, make_plan
    model = CFTN(Config.tiny())
    math = example('math', 5, 'en')
    state = train(model, [math], make_plan('specialist', ('math',), [math]), 1)
    snapshots = {i: {k: v.clone() for k,v in values.items() if torch.is_tensor(v)}
                 for i,values in state['optimizer']['state'].items()}
    string = example('string', 8, 'ro')
    train(model, [string], make_plan('specialist', ('string',), [string]), 1)
    assert all(torch.equal(v, state['optimizer']['state'][i][k]) for i,values in snapshots.items() for k,v in values.items())


def test_specialist_request_enters_computation():
    model = CFTN(Config.tiny()).eval()
    tower = model.towers['math']
    receiver = model.bridges['math']['receiver']
    with torch.no_grad(): receiver.output.weight.normal_(std=.02)
    ids = model.ids('2+2', 'math')
    a = tower.hidden(ids, message=torch.zeros(1,2,tower.width), receiver=receiver)
    b = tower.hidden(ids, message=torch.ones(1,2,tower.width), receiver=receiver)
    assert not torch.equal(a,b)
