"""Read-only local specialist dashboard. No model loading or GPU allocation."""
import argparse
import json
import time
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

def process_alive(pid):
    if not pid:return False
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:return False
        code=wintypes.DWORD()
        try:return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(int(pid),0);return True
    except OSError:return False


def snapshot(root):
    """Read saved evidence without importing a model or touching the GPU."""
    errors=[]
    def read(path):
        try:return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        except (OSError,ValueError) as exc:
            errors.append(f'{path.name}: {exc}');return {}
    queue=read(root/'queue.json')
    if queue.get('output'):
        candidate=Path(queue['output']).resolve()
        if candidate.is_relative_to(root.resolve()):root=candidate
    reports=[read(f) for f in sorted(root.glob('*_epoch_*.json'),key=lambda p:p.stat().st_mtime)]
    reports=[r for r in reports if 'active' in r and 'retention' in r]
    status=read(root/'status.json')
    alive=process_alive(status.get('pid'))
    if status.get('state') in ('training','remediating','evaluating','starting','evaluated') and not alive:
        status={**status,'state':'failed','error':'Training worker is no longer running. Saved results are shown; inspect the worker log before resuming.'}
    latest=next((r for r in reversed(reports) if r.get('phase')==status.get('phase')),reports[-1] if not status.get('phase') and reports else {})
    for key in ('epoch','loss','phase'):
        if key not in status and key in latest:status[key]=latest[key]
    checkpoint=root/'current.specialist'
    return {'status':status,'overfit':read(root/'overfit_test.json'),'queue':queue,
        'curriculum':read(root/'curriculum.json'),'before':read(root/(status.get('phase','baseline')+'_before.json')),
        'history':[{**r,'active':{k:v for k,v in r['active'].items() if k!='samples'},
            'retention':{k:v for k,v in r['retention'].items() if k!='samples'}} for r in reports],
        'samples':latest.get('active',read(root/(status.get('phase','baseline')+'_before.json')).get('active',read(root/'baseline.json'))).get('samples',[])[:16],
        'worker_alive':alive,
        'errors':errors,'root':str(root),'server_time':time.time(),
        'checkpoint':{'path':str(checkpoint),'exists':checkpoint.exists(),
            'updated':checkpoint.stat().st_mtime if checkpoint.exists() else None},
        'tests':{p.name:read(p) for p in root.glob('*_test.json') if p.name!='overfit_test.json'}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8792);a=p.parse_args();root=Path(a.root)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/api':
                payload=json.dumps(snapshot(root)).encode();kind='application/json'
            else:payload=Path(__file__).with_name('local_math_dashboard.html').read_bytes();kind='text/html; charset=utf-8'
            self.send_response(200);self.send_header('Cache-Control','no-store');self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def log_message(self,*args):pass
    ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()


if __name__=='__main__':main()
