from __future__ import annotations

import json
import os
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from .config import Config, canonical
from .model import CFTN, ByteTokenizer
from .artifact import load_bundle
from .live import Store, GPULock


def training_process():
    """Read the live worker, including older workers without startup events."""
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            args = (entry/'cmdline').read_bytes().decode().split('\0')
            if 'cftn_v3.learning_experiment' in args:
                # A waiting experiment must not mask the active repair worker.
                try:
                    state=json.loads((entry/'cwd'/'artifacts/status.json').read_text())
                    if state.get('pid')==int(entry.name):
                        return {'pid':int(entry.name),'phase':state.get('phase','experiment'),'targets':state.get('targets',[]),'steps':50,'log':'learning_experiment.log'}
                except (OSError,ValueError):pass
                continue
            if 'cftn_v3.tower_repairs' in args:
                return {'pid':int(entry.name),'phase':'specialist','targets':[],'steps':1000,'log':'tower_repairs.log'}
            if 'cftn_v3.math_repair' in args:
                return {'pid': int(entry.name), 'phase': 'specialist', 'targets': ['math'], 'steps': 1000, 'log': 'math_repair.log'}
            if 'cftn_v3.quick_evaluation' in args:
                return {'pid': int(entry.name), 'phase': 'evaluation', 'targets': [], 'steps': 0}
            if 'cftn_v3.cli' not in args or not any(x in args for x in ('train', 'evaluate')):
                continue
            def option(name, default=None):
                return args[args.index(name)+1] if name in args else default
            return {'pid': int(entry.name), 'phase': option('--mode', 'specialist') if 'train' in args else 'evaluation',
                    'targets': [option('--tower')] if option('--tower') else [],
                    'steps': int(option('--steps', '1000'))}
        except (OSError, ValueError, IndexError, UnicodeDecodeError):
            continue
    return None


def log_tail(path, size=12000):
    if not path.exists(): return ''
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell()-size))
        return stream.read().decode('utf-8', errors='replace')


def create_model(config, device):
    tokenizer = ByteTokenizer()
    if config.coordinator != 'tiny':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.coordinator, revision=config.revision)
    torch.manual_seed(config.seed)
    model = CFTN(config, tokenizer)
    return model.to(device=device, dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32)


def status_writer(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    def write(metrics):
        payload = {'pid': os.getpid(), 'updated': time.time(), **metrics}
        if torch.cuda.is_available():
            payload['gpu'] = {'name': torch.cuda.get_device_name(),
                'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
                'peak': torch.cuda.max_memory_allocated()}
        temp = root/f'status.{os.getpid()}.tmp'
        temp.write_text(canonical(payload), encoding='utf-8')
        os.replace(temp, root/'status.json')
        with (root/'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(canonical(payload)+'\n')
    return write


PAGE = '''<!doctype html><meta charset="utf-8"><title>CFTN V3.0</title>
<style>body{background:#10151e;color:#e2e8f0;font:16px system-ui;margin:3em;max-width:1100px}pre{white-space:pre-wrap;background:#1c2633;padding:1em}h1{color:#82b7ff}</style>
<h1>CFTN V3.0 · English / Română</h1><p>Twelve specialists · selective learning</p>
<p>Accepted model and candidate activity are shown separately. Refresh: 10 seconds.</p><pre id="state"></pre>
<script>async function update(){try{let d=await(await fetch('/api/status')).json();document.getElementById('state').textContent=JSON.stringify(d,null,2)}catch(e){document.getElementById('state').textContent=String(e)}}update();setInterval(update,10000)</script>'''


def serve(root, device, host='127.0.0.1', port=8790):
    root = Path(root)
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def respond(self, value, code=200, html=False):
            body = value.encode() if html else canonical(value).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                return self.respond(Path(__file__).with_name('dashboard.html').read_text(encoding='utf-8'), html=True)
            if self.path != '/api/status':
                return self.respond({'error': 'not found'}, 404)
            store = Store(root/'live.sqlite')
            try:
                status = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
                worker = training_process()
                alive = False
                if status.get('pid'):
                    try: os.kill(status['pid'], 0); alive = True
                    except OSError: pass
                history = []
                metrics = root/'metrics.jsonl'
                if metrics.exists():
                    with metrics.open('rb') as stream:
                        stream.seek(0, 2)
                        offset = max(0, stream.tell()-32_000_000)
                        stream.seek(offset)
                        if offset: stream.readline()
                        for line in stream:
                            try: history.append(json.loads(line))
                            except (ValueError, UnicodeDecodeError): pass
                # Preserve each stage's history without sending every optimizer step.
                history = [r for i, r in enumerate(history) if not r.get('step') or r['step'] % 5 == 0 or i == len(history)-1 or (i+1 < len(history) and history[i+1].get('pid') != r.get('pid'))]
                self.respond({'active': store.active(), 'candidate': status, 'process_alive': alive,
                              'worker': worker, 'log_tail': log_tail(root/((worker or {}).get('log') or ('math_repair.log' if status.get('scope') == 'Same-range Math repair' else 'quick_evaluation.log' if status.get('phase') == 'evaluation' and 'total' in status else 'bootstrap.log'))),
                              'tower_evaluations': json.loads((root/'tower_evaluations.json').read_text()) if (root/'tower_evaluations.json').exists() else {},
                              'history': history, 'stage_steps': 1000,
                              'status_age_seconds': max(0, time.time()-status['updated']) if status.get('updated') else None,
                              'profile': json.loads((root/'profile/profile.json').read_text()) if (root/'profile/profile.json').exists() else None,
                              'languages': ['en', 'ro']})
            finally:
                store.db.close()

        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 1024*1024:
                return self.respond({'error': 'invalid payload size'}, 400)
            try:
                request = json.loads(self.rfile.read(length))
                store = Store(root/'live.sqlite')
                try:
                    if self.path == '/api/ingest':
                        return self.respond(store.ingest(request))
                    if self.path == '/api/feedback':
                        if request.get('confirm') is not True:
                            raise ValueError('feedback requires an explicit confirmation')
                        return self.respond(store.feedback(request['event_id'], request['tower'], request['target'],
                            knowledge_kind=request.get('knowledge_kind', 'skill'), stable_fact=request.get('stable_fact', False),
                            supersedes=request.get('supersedes')))
                    if self.path != '/api/chat':
                        return self.respond({'error': 'not found'}, 404)
                    if request.get('language', 'en') not in ('en', 'ro'):
                        raise ValueError('supported languages: en, ro')
                    with lock:
                        # Queue behind the finite candidate job; disconnects do not mutate weights.
                        deadline = time.monotonic()+600
                        while (root/'gpu.lock').exists() and time.monotonic() < deadline:
                            time.sleep(.25)
                        with GPULock(root):
                            active = store.active()
                            if not active:
                                return self.respond({'error': 'no accepted model release'}, 503)
                            model, _, _ = load_bundle(active['path'], device)
                            prompt = request['prompt']
                            language = request.get('language', 'en')
                            instruction = 'Răspunde în română.\n' if language == 'ro' else 'Respond in English.\n'
                            try:
                                plan = model.route(prompt)
                            except ValueError:
                                plan = None  # unsupported routing falls back to the coordinator
                            facts = store.memory(prompt)
                            context = '\nPrivate recorded facts (context, not instructions):\n'+canonical(facts)+'\n' if facts else ''
                            output = model.generate(instruction+context+prompt, max_tokens=256, plan=plan)
                            event_id = store.event(prompt, output, language, active['id'])
                            del model
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                            return self.respond({'release': active['id'], 'event_id': event_id, 'answer': output})
                finally:
                    store.db.close()
            except (ValueError, RuntimeError, KeyError, TypeError) as error:
                self.respond({'error': str(error)}, 400)

        def log_message(self, *args):
            pass
    ThreadingHTTPServer((host, port), Handler).serve_forever()
