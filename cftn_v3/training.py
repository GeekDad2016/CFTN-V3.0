from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import TOWERS
from .contracts import UpdatePlan, Evidence, Call, ExecutionPlan
from .data import sample_update
from .teacher_cycles import verify


def parameter_hashes(model):
    return {name: hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters()}


def configure_update(model, plan):
    plan.validate()
    model.eval()
    params = []
    for name, parameter in model.named_parameters():
        parameter.grad = None
        parameter.requires_grad_(plan.allows(name))
        if parameter.requires_grad:
            params.append(parameter)
    for target in plan.targets:
        if plan.mode in {'specialist', 'continual', 'integration'}:
            model.towers[target].train()
    if not params:
        raise ValueError('update selects no parameters')
    return params


def supervised_loss(model, row, tower=None, messages=()):
    tokenizer = model.tokenizer_for(tower) if tower else model.tokenizer
    prefix = tokenizer.encode(row['prompt']+'\n', add_special_tokens=False)
    target = tokenizer.encode(row['target'], add_special_tokens=False)+[tokenizer.eos_token_id]
    device = next(model.parameters()).device
    ids = torch.tensor([prefix+target], device=device)
    if tower:
        logits = model.towers[tower](ids[:, :-1])
    else:
        logits = model.coordinator.logits(model.coordinator.hidden(ids[:, :-1], messages))
    # Prompt tokens are context, never supervised targets.
    return F.cross_entropy(logits[:, len(prefix)-1:].reshape(-1, logits.shape[-1]).float(),
                           ids[:, len(prefix):].reshape(-1))


def routing_loss(model, row):
    feature = model.coordinator.stable_features(model.ids(row['prompt']))
    wakes, rounds, halt = model.dispatcher(feature)
    target = torch.zeros_like(wakes)
    labels = row.get('routing', {'targets': [row['tower']], 'rounds': {row['tower']: 0}})
    round_loss = wakes.sum()*0
    for name in labels['targets']:
        i = TOWERS.index(name)
        target[0, i] = 1
        round_loss += F.cross_entropy(rounds[:, i], torch.tensor([labels['rounds'][name]], device=wakes.device))
    deps = model.dispatcher.dependencies(feature)
    dep_target = torch.zeros_like(deps)
    for name, required in labels.get('dependencies', {}).items():
        for parent in required:
            dep_target[0,TOWERS.index(name),TOWERS.index(parent)] = 1
    return F.binary_cross_entropy_with_logits(wakes, target)+round_loss+F.binary_cross_entropy_with_logits(halt, torch.ones_like(halt))+F.binary_cross_entropy_with_logits(deps, dep_target)


def make_plan(mode, targets, rows, *, verifier=verify):
    evidence = tuple(Evidence(r['id'], r['tower'], verifier(r), r.get('verifier', '')) for r in rows)
    return UpdatePlan(mode, tuple(targets), evidence).validate()


def train(model, rows, plan, steps, *, replay=(), status=None, state=None, verifier=verify):
    if not rows or any(not verifier(r) for r in rows):
        raise ValueError('training requires verified targets')
    if any(r['tower'] not in plan.targets for r in rows):
        raise ValueError('data exceeds update authorization')
    limit = model.config.max_continual_steps if plan.mode == 'continual' else model.config.max_acquisition_steps
    if not 1 <= steps <= limit:
        raise ValueError('training step budget exceeded')
    before = parameter_hashes(model)
    configure_update(model, plan)
    groups = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            lr = model.config.specialist_lr
            if plan.mode == 'continual': lr = model.config.continual_lr
            elif name.startswith('coordinator.adapter.'): lr = model.config.adapter_lr
            elif not name.startswith('towers.'): lr = model.config.bridge_lr
            groups.append({'params': [parameter], 'lr': lr, 'initial_lr': lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    start = 0
    if state:
        if state['mode'] != plan.mode or tuple(state['targets']) != plan.targets:
            raise ValueError('optimizer update-plan mismatch')
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        if torch.cuda.is_available() and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    if start+steps > limit:
        raise ValueError('resumed update exceeds the stage step budget')
    device = next(model.parameters()).device
    batch = model.config.effective_batch
    begun = time.time()
    losses = []
    for step in range(start, start+steps):
        selected = sample_update(rows, replay, batch, model.config.seed+step)
        optimizer.zero_grad(set_to_none=True)
        factor = min(1., (step+1)/max(1, min(100, limit//100)))
        factor *= max(.1, 1-step/limit)
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr']*factor
        value = 0.
        for row in selected:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                if plan.mode == 'routing':
                    loss = routing_loss(model, row)
                elif plan.mode == 'planning':
                    loss = supervised_loss(model, row)
                elif plan.mode in {'specialist', 'continual'}:
                    loss = supervised_loss(model, row, row['tower'])
                else:
                    route = row.get('routing', {'targets': [row['tower']], 'rounds': {row['tower']: 0}})
                    calls = [Call(t, route['rounds'][t], route.get('requests',{}).get(t,row['prompt']), tuple(route.get('dependencies',{}).get(t,()))) for t in route['targets']]
                    execution = ExecutionPlan(calls).validate(TOWERS)
                    messages = model.communicate(row['prompt'], execution)
                    loss = supervised_loss(model, row, messages=messages)
                    if plan.mode == 'integration':
                        if row.get('specialist_targets'):
                            for name, local in row['specialist_targets'].items():
                                loss = loss+supervised_loss(model, local, name)
                        else:
                            loss = loss+supervised_loss(model, row, row['tower'])
                (loss/batch).backward()
                value += float(loss.detach())/batch
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        losses.append(value)
        if status:
            status({'phase': plan.mode, 'step': step+1, 'loss': value,
                    'targets': plan.targets, 'elapsed_seconds': time.time()-begun,
                    'examples_per_second': batch*(step-start+1)/max(.01,time.time()-begun)})
    after = parameter_hashes(model)
    leaked = [name for name in before if not plan.allows(name) and before[name] != after[name]]
    if leaked:
        raise RuntimeError(f'update isolation violated: {leaked}')
    return {'mode': plan.mode, 'targets': plan.targets, 'step': start+steps,
            'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
            'python_rng': random.getstate(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'loss': losses[-1], 'frozen_hashes_verified': True,
            'changed': [name for name in before if before[name] != after[name]]}
