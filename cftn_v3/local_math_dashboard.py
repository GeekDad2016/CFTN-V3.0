"""Read-only local specialist dashboard. No model loading or GPU allocation."""
import argparse
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

PAGE='''<!doctype html><html lang="en"><meta charset="utf-8"><title>CFTN · Local Maths</title>
<style>body{background:#101827;color:#e2e8f0;font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}h1{font-size:30px}section{background:#1e293b;padding:22px;border-radius:12px;margin:20px 0}strong{color:#5eead4}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px monospace}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #334155}.muted{color:#94a3b8}</style>
<h1>CFTN · Local Maths training</h1><p class="muted">RTX 4070 · One specialist · Coordinator and other towers are not loaded</p>
<section><h2 id="state">Connecting…</h2><p id="progress"></p><p id="details"></p></section>
<section><h2>Learning checks</h2><p id="check"></p><table><thead><tr><th>Phase / epoch</th><th>Held-out answers</th><th>Worked traces</th><th>Retention</th></tr></thead><tbody id="history"></tbody></table></section>
<section><h2>Latest evaluation examples</h2><div id="examples"></div></section>
<p class="muted">Acceptance requires two passing validation checks, a separate test, and retention. Training pauses on failure rather than running indefinitely.</p>
<script>async function refresh(){try{const d=await(await fetch('/api')).json(),s=d.status;document.getElementById('state').textContent=(s.state||'Waiting').replaceAll('_',' ');document.getElementById('progress').textContent=`Phase: ${s.phase||'baseline'} | Epoch: ${s.epoch||0}/${s.epochs||30} | Step: ${s.step||0}`;document.getElementById('details').textContent=s.error||s.reason||`Loss: ${s.loss??'—'} | Last update: ${s.updated?new Date(s.updated*1000).toLocaleTimeString():'—'}`;
const probe=d.overfit;document.getElementById('check').textContent=probe?`Disposable overfit test: ${Math.round(probe.before.accuracy*100)}% to ${Math.round(probe.after.accuracy*100)}% correct. Production weights were preserved.`:'First checking the existing model and whether optimization can fit a small sample.';
let host=document.getElementById('history');host.replaceChildren();for(const r of d.history){let tr=document.createElement('tr');for(const v of [r.phase+' / '+r.epoch,Math.round(r.active.accuracy*100)+'%',Math.round(r.active.trace_accuracy*100)+'%',Math.round(r.retention.accuracy*100)+'%']){let td=document.createElement('td');td.textContent=v;tr.append(td)}host.append(tr)}
host=document.getElementById('examples');host.replaceChildren();for(const r of d.samples){let pre=document.createElement('pre');pre.textContent='Request: '+r.prompt+'\nExpected: '+r.expected+'\nModel: '+r.output+'\nCorrect answer: '+r.answer_correct;host.append(pre)}}catch(e){document.getElementById('state').textContent='Dashboard unavailable: '+e.message}}refresh();setInterval(refresh,5000);</script></html>'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8792);a=p.parse_args();root=Path(a.root)
    def read(path):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/api':
                files=sorted(root.glob('*_epoch_*.json'),key=lambda p:p.stat().st_mtime)[-10:]
                reports=[read(f) for f in files]
                samples=(reports[-1]['active'] if reports else read(root/'baseline.json')).get('samples',[])[:8]
                payload=json.dumps({'status':read(root/'status.json'),'overfit':read(root/'overfit_test.json'),
                    'history':[{**r,'active':{k:v for k,v in r['active'].items() if k!='samples'},
                                'retention':{k:v for k,v in r['retention'].items() if k!='samples'}} for r in reports],
                    'samples':samples}).encode();kind='application/json'
            else:payload=PAGE.encode();kind='text/html; charset=utf-8'
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def log_message(self,*args):pass
    ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()


if __name__=='__main__':main()
