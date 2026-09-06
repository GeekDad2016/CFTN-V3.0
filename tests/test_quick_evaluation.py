from cftn_v3 import quick_evaluation as q
from cftn_v3.data import example


def test_resume_and_checkpoint_change(tmp_path, monkeypatch):
    calls=[]
    def evaluate(model, rows, **kwargs):
        r=rows[0]; calls.append(r['id'])
        return {'outputs':[dict(id=r['id'], tower=r['tower'], language='en', output=r['target'], correct=True)]}
    monkeypatch.setattr(q,'evaluate',evaluate)
    rows=[example('math',i,'en') for i in range(3)]+[example('math',0,'ro')]
    q.run(None,rows,tmp_path,'checkpoint1',lambda s:None,count=2)
    assert len(calls)==2
    q.run(None,rows,tmp_path,'checkpoint1',lambda s:None,count=2)
    assert len(calls)==2
    q.run(None,rows,tmp_path,'checkpoint2',lambda s:None,count=2)
    assert len(calls)==4
