"""Pinned English reference-checked distillation, separate from public ingestion."""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .config import Config, canonical, identity
from .data import file_hash, read_rows, verify as builtin_verify

REPO = 'openai/gsm8k'
REVISION = '740312add88f781978c0658806c59bc2815b9866'
SOURCE = 'hf_gsm8k_reference_v1'


def number(value):
    value = value.strip().replace(',', '')
    if not re.fullmatch(r'-?\d+(?:\.\d+)?', value):
        raise ValueError('expected a finite decimal')
    return Decimal(value)


def reference_row(raw, split, index):
    gold = str(number(raw['answer'].rsplit('####', 1)[1]))
    semantic = identity(raw['question'].strip())
    row = dict(source=SOURCE, source_revision=REVISION, source_split=split,
        source_index=index, language='en', tower='math', criterion='english_math_word_problem',
        semantic_id=semantic, prompt=raw['question'].strip()+'\nReturn only <answer>NUMBER</answer>.',
        target=f'<answer>{gold}</answer>', gold=gold, verifier='pinned_reference_answer_v1',
        verified=True, routing={'targets': ['math'], 'rounds': {'math': 0}})
    row['id'] = identity([SOURCE, REVISION, split, index, semantic])
    return row


def prepare(root):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest.json').exists():
        catalog(str(root.resolve()))
        return json.loads((root/'manifest.json').read_text())
    seen, files = set(), {}
    for split in ('test', 'train'):
        path = hf_hub_download(REPO, f'main/{split}-00000-of-00001.parquet',
            repo_type='dataset', revision=REVISION, local_dir=root/'source')
        rows = []
        for index, raw in enumerate(pq.read_table(path).to_pylist()):
            row = reference_row(raw, split, index)
            if row['semantic_id'] in seen: continue
            seen.add(row['semantic_id'])
            rows.append(row)
        out = root/f'{split}.jsonl'
        out.write_text(''.join(canonical(r)+'\n' for r in rows), encoding='utf-8')
        files[split] = {'sha256': file_hash(out), 'records': len(rows)}
    manifest = {'repo': REPO, 'revision': REVISION, 'license': 'MIT', 'files': files,
        'verification': 'Final numeric answer against pinned dataset reference; no reasoning validation.',
        'scope': 'Math tower only; test split is never sent to teacher or training.'}
    (root/'manifest.json').write_text(canonical(manifest), encoding='utf-8')
    return manifest


@functools.lru_cache(maxsize=4)
def catalog(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['repo'] != REPO or manifest['revision'] != REVISION:
        raise ValueError('unsupported teacher source revision')
    records = {}
    for split in ('train', 'test'):
        path = root/f'{split}.jsonl'
        if file_hash(path) != manifest['files'][split]['sha256']:
            raise ValueError('teacher source checksum mismatch')
        records.update({r['id']: r for r in read_rows(path)})
    return records


def verify(row, output=None):
    if row.get('source') != SOURCE:
        return builtin_verify(row, output)
    try:
        original = catalog(str(Path(os.environ.get('CFTN_TEACHER_DATA', 'data/teacher')).resolve()))[row['id']]
        for field in ('prompt', 'source_revision', 'source_split', 'semantic_id', 'tower', 'language', 'verifier'):
            if row.get(field) != original[field]: return False
        answer = row.get('target', '') if output is None else output
        match = re.fullmatch(r'\s*<answer>\s*([^<>]+)\s*</answer>\s*', answer)
        return bool(match and number(match[1]) == number(original['gold']))
    except (OSError, KeyError, ValueError, InvalidOperation):
        return False


def import_verified(store, rows):
    """Only called by the offline worker; public ingest still strips trust claims."""
    for row in rows:
        if row.get('source') != SOURCE or row.get('source_split') != 'train' or not verify(row):
            raise ValueError('unverified or held-out teacher record')
    with store.db:
        for row in rows:
            store.db.execute('INSERT OR IGNORE INTO interactions(id,payload,created,verified) VALUES(?,?,?,1)',
                (row['id'], canonical(row), time.time()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'cycle'])
    p.add_argument('--root', default='artifacts')
    p.add_argument('--data', default='data')
    p.add_argument('--limit', type=int, default=128)
    p.add_argument('--steps', type=int, default=100)
    args = p.parse_args()
    source = Path(args.data)/'teacher'
    os.environ['CFTN_TEACHER_DATA'] = str(source.resolve())
    if args.command == 'prepare':
        print(canonical(prepare(source)))
        return
    if not 32 <= args.limit <= 256 or not 1 <= args.steps <= 200:
        raise ValueError('cycle bounds: 32-256 questions, 1-200 optimizer steps')
    from .live import Store, GPULock
    from .runtime import status_writer
    from .cli import teacher
    from .learner import learn_once
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    writer = status_writer(root)
    store = Store(root/'live.sqlite')
    try:
        if not store.active():
            print(canonical({'state': 'waiting_for_accepted_bootstrap'}))
            return
        with GPULock(root):
            catalog(str(source.resolve()))
            cursor_row = store.db.execute("SELECT value FROM settings WHERE key='teacher_cursor'").fetchone()
            cursor = int(cursor_row[0]) if cursor_row else 0
            rows = read_rows(source/'train.jsonl')[cursor:cursor+args.limit]
            if not rows:
                print(canonical({'state': 'teacher_dataset_exhausted'}))
                return
            incoming, verified = source/'cycle_input.jsonl', source/'cycle_verified.jsonl'
            incoming.write_text(''.join(canonical(r)+'\n' for r in rows), encoding='utf-8')
            writer({'state': 'generating', 'phase': 'teacher', 'targets': ['math'], 'questions': len(rows)})
            teacher_args = argparse.Namespace(input=str(incoming), output=str(verified), tiny=False,
                device='cuda', limit=None, root=str(root))
            report = teacher(teacher_args, Config())
            import_verified(store, read_rows(verified))
            with store.db:
                store.db.execute("INSERT OR REPLACE INTO settings VALUES('teacher_cursor',?)", (str(cursor+len(rows)),))
            learn_args = argparse.Namespace(root=str(root), data=args.data, tower='math', force=False,
                steps=args.steps, device='cuda')
            result = learn_once(learn_args, writer)
            writer({'state': 'finished', 'phase': 'teacher_cycle', 'result': result, 'teacher': report})
            print(canonical(result))
    finally:
        store.db.close()


if __name__ == '__main__': main()
