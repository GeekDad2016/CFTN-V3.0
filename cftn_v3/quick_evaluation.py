"""Resumable English progress panel. Never activates a release."""
import argparse
import json
import time
from pathlib import Path
from .config import TOWERS, canonical, identity
from .data import read_rows, file_hash
from .evaluation import evaluate


def run(model, rows, root, checkpoint_hash, writer, count=32):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    panel = [r for t in TOWERS for r in
             [x for x in rows if x['tower'] == t and x['language'] == 'en' and not x.get('specialist_targets')][:count]]
    key = identity([checkpoint_hash, panel, 128, 719, 'native-progress-v1'])
    path = root/'quick_evaluation.json'
    saved = json.loads(path.read_text()) if path.exists() else {}
    outputs = saved.get('outputs', []) if saved.get('identity') == key else []
    if [o['id'] for o in outputs] != [r['id'] for r in panel[:len(outputs)]]:
        raise ValueError('invalid evaluation resume sequence')
    started, initial = time.time(), len(outputs)
    def persist():
        reports = {}
        for tower in TOWERS:
            local = [o for o in outputs if o['tower'] == tower]
            n, correct = len(local), sum(o['correct'] for o in local)
            reports[tower] = {'groups': {f'{tower}:en': {'count': n, 'correct': correct, 'accuracy': correct/n if n else 0}},
                'samples': local[:4], 'scope': 'Informational English panel; not release acceptance',
                'completed': n, 'total': sum(r['tower'] == tower for r in panel)}
        for dest, payload in [(path, {'identity': key, 'outputs': outputs, 'total': len(panel), 'completed': len(outputs)}),
                              (root/'tower_evaluations.json', reports)]:
            tmp = dest.with_suffix('.tmp')
            tmp.write_text(canonical(payload), encoding='utf-8')
            tmp.replace(dest)
    persist()
    for row in panel[len(outputs):]:
        writer({'phase': 'evaluation', 'state': 'evaluating', 'targets': [row['tower']],
                'completed': len(outputs), 'total': len(panel), 'panel_size': count})
        result = evaluate(model, [row], max_tokens=128)['outputs'][0]
        outputs.append({**result, 'prompt': row['prompt'], 'expected': row['target']})
        persist()
        elapsed = time.time()-started
        writer({'phase': 'evaluation', 'state': 'evaluating', 'targets': [row['tower']],
                'completed': len(outputs), 'total': len(panel), 'correct': sum(o['correct'] for o in outputs),
                'elapsed_seconds': elapsed, 'remaining_seconds': elapsed/max(1,len(outputs)-initial)*(len(panel)-len(outputs))})
    writer({'phase': 'evaluation', 'state': 'finished', 'completed': len(outputs), 'total': len(panel),
            'correct': sum(o['correct'] for o in outputs), 'scope': 'Informational only; no release activation'})
    return outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', default='artifacts/bootstrap.cftn')
    p.add_argument('--root', default='artifacts')
    p.add_argument('--data', default='data')
    p.add_argument('--count', type=int, default=32)
    args = p.parse_args()
    if not 1 <= args.count <= 500: raise ValueError('panel count out of bounds')
    from .live import GPULock
    from .runtime import status_writer
    from .artifact import load_bundle
    writer = status_writer(args.root)
    with GPULock(args.root):
        try:
            writer({'phase': 'evaluation', 'state': 'starting', 'total': args.count*12})
            digest = file_hash(args.bundle)
            model, _, _ = load_bundle(args.bundle, 'cuda')
            run(model, read_rows(Path(args.data)/'development.jsonl'), args.root, digest, writer, args.count)
        except Exception as exc:
            writer({'phase': 'evaluation', 'state': 'failed', 'error': str(exc)})
            raise


if __name__ == '__main__': main()
