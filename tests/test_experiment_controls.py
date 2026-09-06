import pytest
from cftn_v3.experiment_controls import submit,snapshot,answer_pending,teaching,consumed,control

def test_questions_and_explicit_teaching(tmp_path):
    key=submit(tmp_path,{'tower':'math','prompt':'2+2?','expected':'4','teach':True})
    assert snapshot(tmp_path)[0]['result'] is None
    class Model:
        def generate(self,*args,**kwargs):return '4'
    answer_pending(tmp_path,Model(),'test checkpoint')
    assert snapshot(tmp_path)[0]['result']['exact_expected_match']
    assert teaching(tmp_path,'math')[0]['feedback_id']==key
    consumed(tmp_path,[key])
    assert not teaching(tmp_path,'math')
    assert snapshot(tmp_path)[0]['used_for_learning']
    with pytest.raises(ValueError):submit(tmp_path,{'tower':'math','prompt':'question','teach':True})
    assert control(tmp_path,'pause')['pause_requested']

def test_default_test_is_not_training(tmp_path):
    submit(tmp_path,{'tower':'math','prompt':'2+2?','expected':'4'})
    class Model:
        def generate(self,*args,**kwargs):return '5'
    answer_pending(tmp_path,Model(),'test')
    assert not teaching(tmp_path,'math')
