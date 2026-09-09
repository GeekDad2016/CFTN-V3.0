"""Local specialist dashboard with queued validation control. No model loading or GPU allocation."""
import argparse
import json
import time
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from .file_io import read_text_retry

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
        try:return json.loads(read_text_retry(path)) if path.exists() else {}
        except (OSError,ValueError) as exc:
            errors.append(f'{path.name}: {exc}');return {}
    queue=read(root/'queue.json')
    if queue.get('output'):
        candidate=Path(queue['output']).resolve()
        if candidate.is_relative_to(root.resolve()):root=candidate
    reports=[read(f) for f in sorted(root.glob('*_epoch_*.json'),key=lambda p:p.stat().st_mtime)]
    reports=[r for r in reports if 'active' in r and 'retention' in r]
    status=read(root/'status.json')
    fallback=read(root/'status_fallback.json')
    if fallback.get('updated',0)>status.get('updated',0):status=fallback
    if status.get('status_warning'):errors.append(status['status_warning'])
    alive=process_alive(status.get('pid'))
    if status.get('state') in ('training','remediating','evaluating','starting','evaluated') and not alive:
        status={**status,'state':'failed','error':'Training worker is no longer running. Saved results are shown; inspect the worker log before resuming.'}
    latest=next((r for r in reversed(reports) if r.get('phase')==status.get('phase')),reports[-1] if not status.get('phase') and reports else {})
    for key in ('epoch','loss','phase'):
        if key not in status and key in latest:status[key]=latest[key]
    if latest.get('epoch')==status.get('epoch') and latest.get('controller'):
        status={**status,'consolidation_done':latest['controller'].get('consolidation_done',status.get('consolidation_done',0))}
    checkpoint=root/'current.specialist'
    before=read(root/(status.get('phase','baseline')+'_before.json'))
    promotion_path=root/(status.get('phase','baseline')+'_promotion_validation.json')
    promotion=read(promotion_path)
    # A routine panel must never overwrite the evidence used for promotion.
    observation=promotion or latest or before
    evidence_kind=('Full experiment evaluation' if status.get('experiment_arm') else 'Full promotion check') if promotion else 'Routine validation' if latest else 'Stage baseline'
    evidence_round=promotion.get('epoch') if promotion else latest.get('epoch')
    if promotion and evidence_round is None and promotion_path.exists():
        prior_reports=[r for r in reports if r.get('phase')==status.get('phase') and
            (root/f"{r['phase']}_epoch_{r['epoch']:03d}.json").stat().st_mtime<=promotion_path.stat().st_mtime]
        evidence_round=prior_reports[-1]['epoch'] if prior_reports else None
    manual=read(root/(status.get('phase','baseline')+'_manual_validation.json'))
    if manual and manual.get('epoch',-1)>=max(promotion.get('epoch',-1) or -1,latest.get('epoch',-1) or -1):
        observation=manual;evidence_kind='Manual validation (routine panel)';evidence_round=manual['epoch']
    manual_records={}
    for path in list(root.glob('*_manual_round_*.json'))+list(root.glob('*_manual_validation.json')):
        record=read(path)
        if record.get('phase') and record.get('epoch') is not None:manual_records[(record['phase'],record['epoch'])]=record
    manual_history=sorted(manual_records.values(),key=lambda r:r.get('updated',0))
    criterion_details=[]
    for kind in ('active','retention'):
        report=observation.get(kind,{})
        for name,metrics in report.get('criteria',{}).items():
            minimum=max(.95,before.get('retention',{}).get('criteria',{}).get(name,{}).get('accuracy',0)) if kind=='retention' and not promotion else .95
            passed=bool(metrics.get('examples',0) and metrics.get('accuracy',0)>=minimum
                and metrics.get('format_accuracy',0)>=.95 and (kind=='retention' or metrics.get('trace_accuracy',0)>=.90))
            passed=passed and all(m.get('accuracy',0)>=.95 and m.get('format_accuracy',0)>=.95
                and (kind=='retention' or m.get('trace_accuracy',0)>=.90) for m in metrics.get('strata',{}).values())
            criterion_details.append({'name':name,'kind':kind,'metrics':metrics,'passed':passed,
                'answer_threshold':minimum,'round':evidence_round,'evidence_kind':evidence_kind,
                'samples':[r for r in report.get('samples',[]) if r.get('criterion')==name]})
    return {'status':status,'overfit':read(root/'overfit_test.json'),'queue':queue,
        'display_evaluation':{'kind':evidence_kind,'epoch':evidence_round,
            'active':{k:v for k,v in observation.get('active',{}).items() if k!='samples'},
            'retention':{k:v for k,v in observation.get('retention',{}).items() if k!='samples'},
            'passed':observation.get('passed')},
        'routine_evaluation':{k:v for k,v in latest.items() if k in ('epoch','passed')},
        'curriculum':read(root/'curriculum.json'),'before':before,'criterion_details':criterion_details,
        'history':[{**r,'active':{k:v for k,v in r['active'].items() if k!='samples'},
            'retention':{k:v for k,v in r['retention'].items() if k!='samples'}} for r in reports],
        'samples':sorted(observation.get('active',{}).get('samples',[]),key=lambda r:bool(r.get('answer_correct') and r.get('trace_correct')))[:16],
        'manual_history':manual_history,
        'worker_alive':alive,'manual_validation_pending':(root/'VALIDATE_REQUEST.json').exists(),
        'errors':errors,'root':str(root),'server_time':time.time(),
        'checkpoint':{'path':str(checkpoint),'exists':checkpoint.exists(),
            'updated':checkpoint.stat().st_mtime if checkpoint.exists() else None},
        'tests':{p.name:read(p) for p in root.glob('*_test.json') if p.name!='overfit_test.json'}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8792);a=p.parse_args();root=Path(a.root)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path!='/api/validate':self.send_error(404);return
            if self.headers.get('X-CFTN-Action')!='validate':self.send_error(403);return
            data=snapshot(root)
            if not data['worker_alive']:self.send_error(409,'Training worker is not running');return
            target=Path(data['root'])/'VALIDATE_REQUEST.json'
            try:
                with target.open('x') as stream:json.dump({'phase':data['status'].get('phase'),'requested':time.time()},stream)
            except FileExistsError:pass
            self.send_response(202);self.send_header('Content-Type','application/json');self.end_headers()
            self.wfile.write(b'{"queued":true}')
        def do_GET(self):
            if self.path=='/api':
                payload=json.dumps(snapshot(root)).encode();kind='application/json'
            else:payload=Path(__file__).with_name('local_math_dashboard.html').read_bytes();kind='text/html; charset=utf-8'
            self.send_response(200);self.send_header('Cache-Control','no-store');self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def log_message(self,*args):pass
    ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()


if __name__=='__main__':main()
