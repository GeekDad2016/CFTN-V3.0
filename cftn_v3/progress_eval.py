"""Small deterministic English panels; informational, never release gates."""
import json
from pathlib import Path
from .data import read_rows
from .config import canonical
from .evaluation import evaluate


def progress_evaluate(model, data, root, targets, writer, count=32):
    rows = read_rows(Path(data)/'development.jsonl')
    root = Path(root)
    path = root/'tower_evaluations.json'
    reports = json.loads(path.read_text()) if path.exists() else {}
    for tower in targets:
        panel = [r for r in rows if r['tower'] == tower and r['language'] == 'en' and not r.get('specialist_targets')][:count]
        writer({'state': 'evaluating', 'phase': 'evaluation', 'targets': [tower], 'panel_size': len(panel)})
        report = evaluate(model, panel, max_tokens=128)
        report['scope'] = 'Informational English development subset; not a release gate'
        report['samples'] = [{'prompt': r['prompt'], 'expected': r['target'], **o} for r, o in zip(panel[:4], report['outputs'][:4])]
        reports[tower] = report
        temporary = path.with_suffix('.tmp')
        temporary.write_text(canonical(reports), encoding='utf-8')
        temporary.replace(path)
