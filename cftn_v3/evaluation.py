from __future__ import annotations

from collections import defaultdict
import random
import torch

from .config import TOWERS, LANGUAGES
from .teacher_cycles import verify
from .contracts import ExecutionPlan, Call


def evaluate(model, rows, *, seed=719, max_tokens=256, mode='native'):
    torch.manual_seed(seed)
    random.seed(seed)
    model.eval()
    groups = defaultdict(lambda: {'correct': 0, 'count': 0})
    outputs = []
    for row in rows:
        if mode == 'native' and row.get('specialist_targets'):
            continue
        tower = row['tower']
        if mode == 'native':
            output = model.generate(row['prompt']+'\n', tower, max_tokens)
        else:
            route = row.get('routing', {'targets':[tower], 'rounds':{tower:0}})
            plan = ExecutionPlan([Call(t, route['rounds'][t], row['prompt'], tuple(route.get('dependencies',{}).get(t,()))) for t in route['targets']])
            output = model.generate(row['prompt']+'\n', max_tokens=max_tokens,
                plan=plan if mode != 'generalist' else None,
                disabled=(tower,) if mode == 'disabled' else ())
        correct = verify(row, output)
        group = groups[f'{tower}:{row["language"]}']
        group['count'] += 1
        group['correct'] += int(correct)
        outputs.append({'id': row['id'], 'output': output, 'correct': correct,
                        'tower': tower, 'language': row['language']})
    return {'seed': seed, 'mode': mode, 'groups': {
        k: {**g, 'accuracy': g['correct']/g['count']} for k, g in groups.items()}, 'outputs': outputs}


def gate(candidate, baseline=None, *, targets=TOWERS, minimum_per_language=500):
    failures = []
    for tower in targets:
        for language in LANGUAGES:
            key = f'{tower}:{language}'
            observed = candidate['groups'].get(key, {})
            threshold = .99 if tower == 'string' else .90
            if observed.get('count', 0) < minimum_per_language:
                failures.append(f'{key}:insufficient_panel')
            if observed.get('accuracy', 0) < threshold:
                failures.append(f'{key}:acquisition')
            old = (baseline or {}).get('groups', {}).get(key)
            if old and observed.get('accuracy', 0) < old['accuracy']-.01:
                failures.append(f'{key}:retention')
    return {'passed': not failures, 'failures': failures}


def release_gate(reports, baselines, integration, routing, *, targets=TOWERS):
    failures = []
    if len(reports) != 2 or len({r['seed'] for r in reports}) != 2:
        failures.append('two_independent_seeds_required')
    for i, report in enumerate(reports):
        failures.extend(gate(report, baselines[i] if i < len(baselines) else None, targets=targets)['failures'])
    if not integration or integration.get('accuracy_drop', 1) > .01 or integration.get('validity', 0) < .995:
        failures.append('integration_gate')
    if not routing or routing.get('accuracy', 0) < .99 or routing.get('inactive_calls', 1):
        failures.append('routing_gate')
    return {'passed': not failures, 'failures': failures}
