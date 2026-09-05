from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from collections import defaultdict

from .config import TOWERS, LANGUAGES, canonical, identity


def example(tower, index, language):
    """Versioned, bounded bilingual fixtures; not broad capability evidence."""
    rng = random.Random(identity([tower, index]))
    # Unique operands prevent translated/renamed copies of the same arithmetic
    # problem entering different splits. Large held-outs are explicitly extrapolation.
    a, b = index+2, rng.randint(2, 999)
    ro = language == 'ro'
    if tower == 'math':
        prompt = f"{'Calculează' if ro else 'Calculate'} {a}+{b}."
        target = f'<work>{a}+{b}={a+b}</work><answer>{a+b}</answer>'
        criterion = 'integer_addition'
    elif tower == 'string':
        text = f'șțâîă-{index}-ĂÎ' if ro else f'exact-{index}-AB'
        prompt = ('Inversează exact: ' if ro else 'Reverse exactly: ')+text
        target, criterion = text[::-1], 'unicode_reverse'
    elif tower == 'code':
        prompt = (f'Scrie funcția Python solve(x) care adună {a}.' if ro else
                  f'Write Python function solve(x) that adds {a}.')+f' Case {index}.'
        target, criterion = f'def solve(x):\n    return x + {a}', 'pure_add_function'
    elif tower == 'formal_logic':
        prompt = (f'Premise: P({index}); P(x)->Q(x); Q(x)->R(x). Demonstrează R({index}).' if ro else
                  f'Premises: P({index}); P(x)->Q(x); Q(x)->R(x). Prove R({index}).')
        target, criterion = f'P({index});Q({index});R({index})', 'horn_chain'
    elif tower == 'science':
        prompt = (f'Un obiect parcurge {a*b} metri în {b} secunde. Viteza în m/s? Caz {index}.' if ro else
                  f'An object travels {a*b} metres in {b} seconds. Speed in m/s? Case {index}.')
        target, criterion = f'{a} m/s', 'speed_units'
    elif tower in {'retrieval', 'long_context'}:
        text = '\n'.join(f'[D{k}] item-{index}-{k}={b+k}' for k in range(64 if tower == 'long_context' else 4))
        k = rng.randrange(64 if tower == 'long_context' else 4)
        prompt = text+(f'\nCare este valoarea item-{index}-{k}? Citează sursa.' if ro else
                       f'\nWhat is the value of item-{index}-{k}? Cite the source.')
        target, criterion = f'{b+k} [D{k}]', 'supplied_evidence'
    elif tower == 'multilingual':
        prompt = (f'Tradu în engleză: Sunt {index} mere.' if ro else
                  f'Translate into Romanian: There are {index} apples.')
        target = f'There are {index} apples.' if ro else f'Sunt {index} mere.'
        criterion = 'en_ro_quantity_translation'
    elif tower == 'tool_use':
        prompt = (f'Folosește instrumentul add(a,b) pentru {a} și {b}. Caz {index}.' if ro else
                  f'Use tool add(a,b) for {a} and {b}. Case {index}.')
        target, criterion = canonical({'tool': 'add', 'args': {'a': a, 'b': b}}), 'typed_add_call'
    elif tower == 'structured_data':
        prompt = (f'Tabel items(id INTEGER,value INTEGER). SQL pentru value la id={index}.' if ro else
                  f'Table items(id INTEGER,value INTEGER). SQL for value at id={index}.')
        target, criterion = f'SELECT value FROM items WHERE id = {index}', 'sqlite_lookup'
    elif tower == 'information_extraction':
        prompt = (f'Extrage câmpurile id și oraș ca JSON: id={index}; oraș=Iași.' if ro else
                  f'Extract id and city as JSON: id={index}; city=Iași.')
        target, criterion = canonical({'id': index, 'city': 'Iași'}), 'typed_fields'
    elif tower == 'commonsense':
        prompt = (f'Caz {index}: Ana pune o carte pe masă și pleacă. Nimeni nu o mută. Unde este cartea? A: masă B: dulap' if ro else
                  f'Case {index}: Ana puts a book on the table and leaves. Nobody moves it. Where is the book? A: table B: cupboard')
        target, criterion = 'A', 'explicit_object_persistence'
    else:
        raise ValueError('unknown tower')
    row = dict(tower=tower, language=language, prompt=prompt, target=target,
               criterion=criterion, source='cftn_bounded_bilingual_v1',
               semantic_id=identity([tower, index]), index=index,
               verified=True, verifier='deterministic_v1', stable_fact=False,
               routing={'targets': [tower], 'rounds': {tower: 0}})
    row['id'] = identity(row)
    return row


def verify(row, output=None):
    """Verifier dispatch. User assertions cannot self-certify as generated data."""
    output = row.get('target', '') if output is None else output
    if row.get('source') == 'cftn_bilingual_composition_v1':
        expected = composition(int(row['index']), row['language'])
        return row['prompt'] == expected['prompt'] and output.strip() == expected['target']
    if row.get('source') == 'cftn_bounded_bilingual_v1':
        try:
            expected = example(row['tower'], int(row['index']), row['language'])
        except (KeyError, ValueError):
            return False
        if row.get('prompt') != expected['prompt']:
            return False
        if row['tower'] == 'code':
            # Evaluate a deliberately restricted pure-expression language, never Python exec.
            try:
                tree = ast.parse(output)
                if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef): return False
                fn = tree.body[0]
                if fn.name != 'solve' or fn.decorator_list or len(fn.args.args) != 1 or fn.args.args[0].arg != 'x': return False
                if fn.args.defaults or fn.args.kw_defaults or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs: return False
                if len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return) or len(list(ast.walk(tree))) > 64: return False
                def compute(node, x):
                    if isinstance(node, ast.Name) and node.id == 'x': return x
                    if isinstance(node, ast.Constant) and type(node.value) == int and abs(node.value) < 10000000: return node.value
                    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                        left, right = compute(node.left, x), compute(node.right, x)
                        result = left+right if isinstance(node.op, ast.Add) else left-right if isinstance(node.op, ast.Sub) else left*right
                        if abs(result) > 10**15: raise ValueError('expression bound')
                        return result
                    raise ValueError('unsupported code syntax')
                return all(compute(fn.body[0].value, x) == x+int(row['index'])+2 for x in (0, 1, -7, 1009, -319))
            except (SyntaxError, ValueError, RecursionError, TypeError):
                return False
        if row['tower'] == 'structured_data':
            import sqlite3
            db = sqlite3.connect(':memory:')
            try:
                db.execute('CREATE TABLE items(id INTEGER,value INTEGER)')
                index = int(row['index'])
                db.executemany('INSERT INTO items VALUES(?,?)', [(index, 73), (index+1, 99), (index-1, -5)])
                db.set_authorizer(lambda action,*args: sqlite3.SQLITE_OK if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ) else sqlite3.SQLITE_DENY)
                budget = [0]
                def progress():
                    budget[0] += 1
                    return int(budget[0] > 100)
                db.set_progress_handler(progress, 100)
                return db.execute(output).fetchmany(2) == [(73,)]
            except sqlite3.Error:
                return False
            finally:
                db.close()
        if row['tower'] in {'tool_use', 'information_extraction'}:
            try:
                return json.loads(output) == json.loads(expected['target'])
            except (ValueError, TypeError):
                return False
        return output.strip() == expected['target'].strip()
    if row.get('verifier') == 'human_confirmed_v1' and row.get('confirmation_id'):
        return output.strip() == row.get('target', '').strip()
    return False


def composition(index, language):
    math = example('math', index, language)
    answer = str(index+2+random.Random(identity(['math', index])).randint(2, 999))
    prompt = math['prompt']+(' Apoi inversează cifrele rezultatului, fără explicații.' if language == 'ro' else
                            ' Then reverse the result digits, without explanation.')
    row = {**math, 'prompt': prompt, 'target': answer[::-1], 'criterion': 'math_then_string',
           'source': 'cftn_bilingual_composition_v1', 'semantic_id': identity(['composition', index]),
           'routing': {'targets': ['math', 'string'], 'rounds': {'math':0,'string':1},
                       'dependencies': {'string':['math']}},
           'specialist_targets': {'math': {'prompt': math['prompt'], 'target': math['target']},
              'string': {'prompt': ('Inversează exact: ' if language=='ro' else 'Reverse exactly: ')+answer,
                         'target': answer[::-1]}}}
    row['id'] = identity({k:v for k,v in row.items() if k!='id'})
    return row


def prepare(root, train_objects=2048, panel_objects=500):
    root = Path(root)
    if (root/'manifest.json').exists():
        return audit(root)
    root.mkdir(parents=True, exist_ok=True)
    files, seen = {}, set()
    offsets = {'train': 0, 'development': 1000000, 'test': 2000000}
    for split, offset in offsets.items():
        count = train_objects if split == 'train' else panel_objects
        path = root/f'{split}.jsonl'
        with path.open('w', encoding='utf-8', newline='\n') as f:
            for tower in TOWERS:
                for index in range(offset, offset+count):
                    for language in LANGUAGES:
                        row = example(tower, index, language)
                        row['split'] = split
                        f.write(canonical(row)+'\n')
            for index in range(offset, offset+count):
                for language in LANGUAGES:
                    row = composition(index, language)
                    row['split'] = split
                    f.write(canonical(row)+'\n')
        files[split] = {'path': path.name, 'sha256': file_hash(path), 'records': count*26}
    manifest = {'format': 'cftn_v3_data_v1', 'files': files, 'languages': list(LANGUAGES),
                'generator_sha256': file_hash(Path(__file__)),
                'scope': 'bounded fixtures; template generalization and broad competence not established'}
    manifest['sha256'] = identity(manifest)
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return audit(root)


def file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def audit(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    unsigned = dict(manifest)
    digest = unsigned.pop('sha256')
    if identity(unsigned) != digest or file_hash(Path(__file__)) != manifest['generator_sha256']:
        raise ValueError('dataset identity mismatch')
    semantic, prompts, ids, count = {}, {}, set(), 0
    for split, meta in manifest['files'].items():
        if file_hash(root/meta['path']) != meta['sha256']:
            raise ValueError('dataset file mismatch')
        rows = read_rows(root/meta['path'])
        if len(rows) != meta['records']:
            raise ValueError('dataset count mismatch')
        for r in rows:
            if r['id'] in ids or not verify(r) or r['language'] not in LANGUAGES:
                raise ValueError('duplicate or invalid record')
            ids.add(r['id'])
            for key, table in ((r['semantic_id'], semantic), (r['prompt'], prompts)):
                if key in table and table[key] != split:
                    raise ValueError('cross-split leakage')
                table[key] = split
            count += 1
    return {'status': 'passed', 'records': count, 'manifest_sha256': digest}


def balanced_sample(rows, count, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['criterion'], row['language'])].append(row)
    if count and not groups:
        raise ValueError('empty training/replay source')
    keys = sorted(groups)
    rng.shuffle(keys)
    return [rng.choice(groups[keys[i % len(keys)]]) for i in range(count)]


def sample_update(new, replay, count, seed, replay_fraction=0.25):
    rng = random.Random(seed)
    old = round(count*replay_fraction) if replay else 0
    rows = balanced_sample(new, count-old, rng)+balanced_sample(replay, old, rng)
    rng.shuffle(rows)
    return rows


def reservoir(rows, capacity=4096):
    unique = {r['id']: r for r in rows}
    groups = defaultdict(list)
    for row in unique.values():
        groups[(row['criterion'], row['language'])].append(row)
    for group in groups.values():
        group.sort(key=lambda r: identity(r['id']))
    result = []
    while groups and len(result) < capacity:
        for key in sorted(list(groups)):
            if len(result) == capacity:
                break
            result.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
    return result
