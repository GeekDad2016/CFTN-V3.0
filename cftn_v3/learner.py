from __future__ import annotations

import gc
import time
from pathlib import Path
import torch

from .artifact import load_bundle, save_bundle
from .config import TOWERS, canonical
from .data import read_rows, reservoir
from .evaluation import evaluate, release_gate
from .live import Store
from .training import make_plan, train, parameter_hashes


def integration_metrics(model, rows):
    # Exact native reference evaluation and coordinator-only comparison remain separate.
    native = evaluate(model, rows, mode='collaboration')
    count = len(native['outputs'])
    correct = sum(r['correct'] for r in native['outputs'])
    routing_correct = inactive = valid = 0
    for row in rows:
        try:
            plan = model.route(row['prompt'])
            selected = {c.tower for c in plan.calls}
            routing_correct += selected == set(row.get('routing', {'targets': [row['tower']]})['targets'])
            inactive += bool(selected-set(model.config.active))
            valid += 1
        except ValueError:
            pass
    return {'accuracy': correct/max(1, count), 'validity': valid/max(1, count)}, {
        'accuracy': routing_correct/max(1, count), 'inactive_calls': inactive}


def assess_release(model, rows, baseline=None, targets=TOWERS):
    reports = [evaluate(model, rows, seed=s) for s in (719, 1729)]
    baselines = [evaluate(baseline, rows, seed=s) for s in (719, 1729)] if baseline else []
    integration, routing = integration_metrics(model, rows)
    if baseline:
        previous, _ = integration_metrics(baseline, rows)
        integration['accuracy_drop'] = previous['accuracy']-integration['accuracy']
    else:
        integration['accuracy_drop'] = 0 if integration['accuracy'] >= .9 else 1
    gate = release_gate(reports, baselines, integration, routing, targets=targets)
    return {**gate, 'native': reports, 'integration': integration, 'routing': routing}


def learn_once(args, writer):
    root = Path(args.root)
    store = Store(root/'live.sqlite')
    accepted = store.active()
    if not accepted:
        raise ValueError('continual learning requires an accepted bootstrap release')
    targets = [args.tower] if args.tower else TOWERS
    target = next((t for t in targets if store.pending(t, args.force)), None)
    if target is None:
        return {'state': 'waiting_for_verified_examples'}
    new = store.pending(target, args.force)
    from .config import identity
    new_panel_path = root/f'new_panel_{target}.jsonl'
    consumed_ids = [r['id'] for r in new]
    cycle_id = identity([accepted['id'], sorted(consumed_ids)])
    previous_failure = store.db.execute('SELECT value FROM settings WHERE key=?', (f'rejected:{target}',)).fetchone()
    if previous_failure and previous_failure[0] == cycle_id and not args.force:
        return {'state': 'waiting_for_new_evidence_after_rejection', 'target': target}
    if new_panel_path.exists():
        new_panel = read_rows(new_panel_path)
    else:
        # Reserve whole semantic groups before any optimizer update. The held-out
        # records are consumed with the cycle and never added to training replay.
        groups = {}
        for r in new:
            groups.setdefault(r.get('semantic_id', identity(r['prompt'])), []).append(r)
        keys = sorted(groups, key=identity)
        holdout = set(keys[:max(1, len(keys)//4)])
        new_panel = [r for key in holdout for r in groups[key]]
        new = [r for key in keys if key not in holdout for r in groups[key]]
    if not new or len(new_panel) < 8:
        return {'state': 'waiting_for_disjoint_new_material_panel'}
    replay_path = root/f'replay_{target}.jsonl'
    replay = read_rows(replay_path) if replay_path.exists() else [r for r in read_rows(Path(args.data)/'train.jsonl') if r['tower'] == target]
    replay = reservoir([r for r in replay if not r.get('specialist_targets')])
    model, _, _ = load_bundle(accepted['path'], args.device)
    baseline_hashes = parameter_hashes(model)
    plan = make_plan('continual', (target,), new)
    state = train(model, new, plan, min(args.steps, 200), replay=replay, status=writer)
    candidate = root/f'candidate_{time.time_ns()}.cftn'
    save_bundle(candidate, model, training=state, metadata={'accepted': False, 'parent': accepted['id']})
    del state, model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    # Sequential baseline/candidate evaluation avoids holding two full GPU models.
    rows = read_rows(Path(args.data)/'development.jsonl')
    baseline, _, _ = load_bundle(accepted['path'], args.device)
    baseline_reports = [evaluate(baseline, rows, seed=s) for s in (719, 1729)]
    previous, _ = integration_metrics(baseline, rows)
    del baseline
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model, _, _ = load_bundle(candidate, args.device)
    reports = [evaluate(model, rows, seed=s) for s in (719, 1729)]
    integration, routing = integration_metrics(model, rows)
    integration['accuracy_drop'] = previous['accuracy']-integration['accuracy']
    report = release_gate(reports, baseline_reports, integration, routing)
    after = parameter_hashes(model)
    if any(after[k] != v for k, v in baseline_hashes.items() if not plan.allows(k)):
        report['passed'] = False
        report['failures'].append('unrelated_parameter_change')
    # New information needs its own held-out acceptance examples; a training score is insufficient.
    if new_panel:
        if ({r.get('semantic_id', r['id']) for r in new+replay} & {r.get('semantic_id', r['id']) for r in new_panel}
            or {r['prompt'] for r in new+replay} & {r['prompt'] for r in new_panel}):
            report['passed'] = False
            report['failures'].append('new_panel_training_overlap')
        new_result = evaluate(model, new_panel)
        if not new_result['outputs'] or sum(r['correct'] for r in new_result['outputs'])/len(new_result['outputs']) < .9:
            report['passed'] = False
            report['failures'].append('new_material_accuracy')
    (root/'candidate_report.json').write_text(canonical(report))
    if not report['passed']:
        with store.db:
            store.db.execute('INSERT OR REPLACE INTO settings VALUES(?,?)', (f'rejected:{target}',cycle_id))
        return {'state': 'rejected', 'candidate': str(candidate), 'failures': report['failures']}
    # Final deployment bundle omits the optimizer and private interaction rows.
    released = root/f'release_{time.time_ns()}.cftn'
    save_bundle(released, model, metadata={'accepted': True, 'parent': accepted['id'], 'report': report})
    release = store.activate(released, report, consumed=consumed_ids)
    replay_path.write_text(''.join(canonical(r)+'\n' for r in reservoir(replay+new)), encoding='utf-8')
    return {'state': 'accepted', 'release': release, 'target': target}
