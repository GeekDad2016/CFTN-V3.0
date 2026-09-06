"""Persistent dashboard tests and explicit experimental teaching examples."""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from .config import TOWERS,canonical,identity


@contextmanager
def db(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(root/'dashboard.sqlite',timeout=30)
    conn.execute('CREATE TABLE IF NOT EXISTS tests(id TEXT PRIMARY KEY,payload TEXT,created REAL,result TEXT,used INTEGER DEFAULT 0)')
    try:
        with conn:yield conn
    finally:conn.close()


def submit(root,request):
    tower=request.get('tower','math');prompt=request.get('prompt','').strip();expected=request.get('expected','').strip()
    if tower not in (*TOWERS,'coordinator','automatic') or not 1<=len(prompt)<=6000 or len(expected)>3000:raise ValueError('Invalid tower or question size')
    teach=request.get('teach') is True
    if teach and (not expected or tower not in ('math','retrieval','code','formal_logic')):
        raise ValueError('Teaching requires an expected answer and Math, Retrieval, Code or Formal logic')
    payload={'tower':tower,'prompt':prompt,'expected':expected,'teach':teach,
             'teacher_answer':str(request.get('teacher_answer',''))[:3000]}
    key=identity([payload,time.time_ns()])
    with db(root) as conn:
        if conn.execute('SELECT COUNT(*) FROM tests WHERE result IS NULL').fetchone()[0]>=32:raise ValueError('Test queue is full')
        conn.execute('INSERT INTO tests(id,payload,created) VALUES(?,?,?)',(key,canonical(payload),time.time()))
    return key


def snapshot(root):
    with db(root) as conn:
        rows=conn.execute('SELECT id,payload,result,used FROM tests ORDER BY created DESC LIMIT 12').fetchall()
    return [{'id':i,**json.loads(p),'result':json.loads(r) if r else None,'used_for_learning':bool(u)} for i,p,r,u in rows]


def pending(root):
    with db(root) as conn:return conn.execute('SELECT COUNT(*) FROM tests WHERE result IS NULL').fetchone()[0]


def math_examples(root):
    root=Path(root)
    reports=[p for p in root.glob('*_math.json') if p.name.split('_')[0].isdigit()]
    if not reports:return None
    latest=max(reports,key=lambda p:int(p.name.split('_')[0]))
    cycle=int(latest.name.split('_')[0]);report=json.loads(latest.read_text())
    from .data import read_rows
    data=root.resolve().parent.parent/'data/learning_experiment'
    count=len(read_rows(data/'math_train.jsonl'))
    batch=cycle%((count+47)//48)
    taught=read_rows(data/f'teacher_{batch}_math.jsonl')[:4]
    return {'round':cycle+1,'trained_examples':taught,'before':report['before'],'after':report['after']}


def queue_math_examples(root):
    examples=math_examples(root)
    if not examples:raise ValueError('No completed Math learning block yet')
    return [submit(root,{'tower':'math','prompt':r['prompt'],'expected':r['reference'],
                        'teacher_answer':r['target'],'teach':False}) for r in examples['trained_examples']]


def answer_pending(root,model,checkpoint):
    with db(root) as conn:
        rows=conn.execute('SELECT id,payload FROM tests WHERE result IS NULL ORDER BY created LIMIT 8').fetchall()
        for key,payload in rows:
            r=json.loads(payload)
            trace=None
            if r['tower']=='automatic':
                from .delegation import automatic
                trace=automatic(model,r['prompt']);output=trace['answer']
            else:
                output=model.generate(r['prompt']+'\n',None if r['tower']=='coordinator' else r['tower'],max_tokens=128)
            result={'delegation':trace,'answer':output,'checkpoint':checkpoint,'answered':time.time(),
                'exact_expected_match':output.strip()==r['expected'] if r['expected'] else None}
            conn.execute('UPDATE tests SET result=? WHERE id=?',(canonical(result),key))
            conn.commit()


def teaching(root,tower):
    from .learning_experiment import record
    from .config import Config
    with db(root) as conn:rows=conn.execute('SELECT id,payload FROM tests WHERE used=0 AND result IS NOT NULL ORDER BY created').fetchall()
    result=[]
    for key,p in rows:
        r=json.loads(p)
        if r['tower']==tower and r['teach']:
            result.append({**record(tower,r['prompt'],r['expected'],'explicit_user_example'),
                'teacher_revision':Config().revision,'feedback_id':key,'supervision':'user_supplied_not_verified'})
    return result[:16]


def consumed(root,ids):
    with db(root) as conn:conn.executemany('UPDATE tests SET used=1 WHERE id=?',[(i,) for i in ids])


def control(root,action):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    flag=root/'PAUSED'
    if action=='pause':flag.write_text('Pause after the current checkpointed training block.')
    elif action=='resume':
        flag.unlink(missing_ok=True)
        running=False
        for entry in Path('/proc').glob('[0-9]*'):
            try:running |= 'cftn_v3.learning_experiment' in (entry/'cmdline').read_bytes().decode().split('\0')
            except (OSError,UnicodeDecodeError):pass
        if not running:
            import subprocess,sys,os
            project=root.resolve().parent.parent
            with (root.parent/'learning_experiment.log').open('ab') as log:
                subprocess.Popen([sys.executable,'-u','-m','cftn_v3.learning_experiment','--wait'],cwd=project,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,
                    env={**os.environ,'CFTN_BUNDLE_SCRATCH':'/tmp'})
    else:raise ValueError('Unknown experiment action')
    return {'pause_requested':flag.exists(),'message':'Pause takes effect after the current block is saved.' if action=='pause' else 'Learning enabled; waits for repairs if necessary.'}
