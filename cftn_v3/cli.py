from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .config import Config, TOWERS, PROFILES, canonical
from .data import prepare, read_rows, verify, reservoir
from .runtime import create_model, status_writer, serve
from .training import train, make_plan
from .artifact import save_bundle, load_bundle, export_bundle
from .evaluation import evaluate, gate, release_gate
from .live import Store, GPULock


def parser():
    p = argparse.ArgumentParser(description='CFTN V3.0 — English and Romanian')
    p.add_argument('command', choices=['profile', 'prepare', 'train', 'evaluate', 'serve',
        'ingest', 'learn', 'export', 'rollback', 'teacher'])
    p.add_argument('--root', default='artifacts')
    p.add_argument('--data', default='data')
    p.add_argument('--config')
    p.add_argument('--tiny', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--bundle')
    p.add_argument('--output')
    p.add_argument('--input')
    p.add_argument('--tower', choices=TOWERS)
    p.add_argument('--mode', default='specialist', choices=['specialist', 'routing', 'communication', 'integration'])
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int, default=719)
    p.add_argument('--confirm', action='store_true', help='Explicitly confirm private correction targets')
    p.add_argument('--force', action='store_true', help='Flush at least 32 verified interactions')
    p.add_argument('--watch', action='store_true')
    p.add_argument('--activate', action='store_true', help='Run complete bootstrap release gates and activate only on success')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8790)
    return p


def profile(args, config):
    if not args.tiny:
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 79_000_000_000:
            raise RuntimeError('production profiling requires at least 80 GB class VRAM; use --tiny for local checks')
    results = []
    for name in (['tiny'] if args.tiny else ['small', 'medium', 'large']):
        config.profile = name
        try:
            torch.cuda.reset_peak_memory_stats() if args.device.startswith('cuda') else None
            model = create_model(config, args.device)
            from .data import example
            rows = [example('math', 2, 'en'), example('string', 3, 'ro')]
            plan = make_plan('integration', ('math', 'string'), rows)
            state = train(model, rows, plan, 1)
            if not args.tiny:
                # Actual worst-length allocation, not extrapolation from a short smoke.
                from .contracts import UpdatePlan, Evidence
                from .training import configure_update
                stress_plan = UpdatePlan('integration', ('math', 'long_context'),
                    (Evidence('profile-math', 'math', True, 'synthetic_shape'),
                     Evidence('profile-long', 'long_context', True, 'synthetic_shape')))
                params = configure_update(model, stress_plan)
                opt = torch.optim.AdamW(params, lr=1e-5)
                loss = None
                for tower_name, length in [('math', config.context), ('long_context', config.long_context)]:
                    tower = model.towers[tower_name]
                    ids = torch.randint(2, tower.embedding.num_embeddings, (1, length), device=args.device)
                    hidden = tower.hidden(ids)
                    term = tower.logits(hidden[:, -32:]).float().square().mean()
                    loss = term if loss is None else loss+term
                ids = torch.randint(2, model.tokenizer.vocab_size, (1, 2048), device=args.device)
                features = model.coordinator.hidden(ids)
                message = model.bridges['math']['return'](hidden[:, -8:, :model.towers['math'].width])
                final = model.coordinator.receivers[0](features, message)
                loss = loss+model.coordinator.logits(final[:, -32:]).float().square().mean()
                loss.backward()
                opt.step()  # Allocate optimizer state before recording the peak.
                del opt, params, loss, hidden, features, final, message, ids
            peak = torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
            reserved = torch.cuda.max_memory_reserved() if args.device.startswith('cuda') else 0
            total = torch.cuda.get_device_properties(0).total_memory if args.device.startswith('cuda') else 1
            result = {'profile': name, 'peak': peak, 'reserved': reserved,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'passed': max(peak, reserved) < total*.8 if peak else args.tiny,
                      'scope': 'tiny local smoke' if args.tiny else 'resident twelve-tower 4096/16384-token backward and optimizer allocation'}
            # Profiling never exports updated random weights as an accepted model.
            results.append(result)
            del state, model
        except torch.cuda.OutOfMemoryError:
            results.append({'profile': name, 'passed': False, 'reason': 'CUDA OOM'})
        finally:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    Path(args.root).mkdir(parents=True, exist_ok=True)
    Path(args.root, 'profile.json').write_text(canonical(results))
    passed = [r for r in results if r['passed']]
    if passed:
        config.profile = passed[-1]['profile']
        Path(args.root, 'selected_config.json').write_text(canonical(config.as_dict()))
    return results


def teacher(args, config):
    if not args.input:
        raise ValueError('teacher requires --input with verifiable source records')
    rows = read_rows(args.input)
    if args.limit: rows = rows[:args.limit]
    model = create_model(config, args.device) if args.tiny else None
    if not args.tiny:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(config.coordinator, revision=config.revision)
        base = AutoModelForCausalLM.from_pretrained(config.coordinator, revision=config.revision, torch_dtype=torch.bfloat16).to(args.device).eval()
    accepted = []
    for row in rows:
        if args.tiny:
            output = model.generate(row['prompt'], max_tokens=64)
        else:
            messages = [{'role': 'user', 'content': row['prompt']}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(args.device)
            with torch.no_grad(): generated = base.generate(ids, max_new_tokens=512, do_sample=False)
            output = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)
        if verify(row, output):
            accepted.append({**row, 'target': output, 'teacher_revision': config.revision})
    path = Path(args.output or Path(args.root)/'teacher_verified.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(canonical(r)+'\n' for r in accepted), encoding='utf-8')
    return {'accepted': len(accepted), 'rejected': len(rows)-len(accepted), 'output': str(path)}


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    config = Config.load(args.config) if args.config else Config.tiny() if args.tiny else Config()
    if args.command == 'learn' and args.watch:
        import sys
        forwarded = list(argv if argv is not None else sys.argv[1:])
        forwarded.remove('--watch')
        while True:
            try:
                main(forwarded)
            except Exception as error:
                print(f'Learning cycle failed: {error}', flush=True)
            time.sleep(60)
    writer = status_writer(root)
    if args.command == 'prepare':
        result = prepare(args.data, train_objects=8 if args.tiny else 2048, panel_objects=2 if args.tiny else 500)
    elif args.command == 'serve':
        return serve(root, args.device, args.host, args.port)
    elif args.command == 'ingest':
        store = Store(root/'live.sqlite')
        result = [store.ingest(r, confirm=args.confirm) for r in read_rows(args.input)]
    elif args.command == 'rollback':
        result = {'release': Store(root/'live.sqlite').rollback()}
    elif args.command == 'export':
        result = {'sha256': export_bundle(args.bundle, args.output)}
    else:
        with GPULock(root):
            try:
                if args.command == 'profile':
                    result = profile(args, config)
                elif args.command == 'teacher':
                    result = teacher(args, config)
                elif args.command == 'train':
                    writer({'state': 'starting', 'phase': args.mode, 'targets': [args.tower] if args.tower else list(TOWERS), 'stage_steps': args.steps})
                    from .data import audit
                    audit(args.data)
                    rows = read_rows(Path(args.data)/'train.jsonl')
                    if args.input:
                        incoming = read_rows(args.input)
                        heldout = {r['semantic_id'] for split in ('development', 'test') for r in read_rows(Path(args.data)/f'{split}.jsonl')}
                        if any(not verify(r) or r.get('semantic_id') in heldout for r in incoming):
                            raise ValueError('distillation input is unverified or overlaps held-out data')
                        by_id = {r['id']: r for r in rows}
                        by_id.update({r['id']: r for r in incoming})
                        rows = list(by_id.values())
                    targets = (args.tower,) if args.tower else TOWERS
                    if args.mode == 'specialist' and not args.tower:
                        raise ValueError('specialist training requires --tower; bootstrap explicitly iterates all twelve')
                    rows = [r for r in rows if r['tower'] in targets]
                    if args.mode == 'specialist': rows = [r for r in rows if not r.get('specialist_targets')]
                    model, state, metadata = load_bundle(args.bundle, args.device, training=True) if args.bundle else (create_model(config, args.device), None, {})
                    plan = make_plan(args.mode, targets, rows)
                    if state and (state['mode'] != plan.mode or tuple(state['targets']) != plan.targets):
                        state = None
                    state = train(model, rows, plan, args.steps, state=state, status=writer)
                    path = args.output or root/'candidate.cftn'
                    writer({'state': 'saving', 'phase': args.mode, 'targets': list(targets), 'step': state['step'], 'loss': state['loss']})
                    result = {'bundle': str(path), 'sha256': save_bundle(path, model, training=state,
                        metadata={'accepted': False, 'dataset': str(Path(args.data).resolve()), 'mode': args.mode}),
                        'loss': state['loss'], 'isolation': state['frozen_hashes_verified']}
                elif args.command == 'evaluate':
                    model, _, _ = load_bundle(args.bundle, args.device)
                    rows = read_rows(args.input or Path(args.data)/'development.jsonl')
                    if args.tower: rows = [r for r in rows if r['tower'] == args.tower]
                    if args.limit: rows = rows[:args.limit]
                    if args.activate:
                        if args.limit or args.tower or args.input:
                            raise ValueError('activation requires the complete audited development panels')
                        from .data import audit
                        from .learner import assess_release
                        audit(args.data)
                        model.config.active = TOWERS
                        result = assess_release(model, rows)
                        if result['passed']:
                            path = root/f'release_{time.time_ns()}.cftn'
                            summary = {k: v for k, v in result.items() if k != 'native'}
                            save_bundle(path, model, metadata={'accepted': True, 'report': summary})
                            result['release'] = Store(root/'live.sqlite').activate(path, summary)
                        Path(args.output or root/'evaluation.json').write_text(canonical(result))
                        result = {k: v for k, v in result.items() if k != 'native'}
                    else:
                        report = evaluate(model, rows, seed=args.seed)
                        result = gate(report, targets=(args.tower,) if args.tower else TOWERS)
                        Path(args.output or root/'evaluation.json').write_text(canonical({'report': report, 'gate': result}))
                elif args.command == 'learn':
                    from .learner import learn_once
                    result = learn_once(args, writer)
                writer({'state': 'finished', 'result': result})
            except Exception as error:
                writer({'state': 'failed', 'error': str(error)})
                raise
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
