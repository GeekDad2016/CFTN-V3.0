from cftn_v3.criterion_repair import failures
from cftn_v3.local_math_dashboard import snapshot
import json

def test_strict_rejects_any_error_including_retained_traces():
    perfect=dict(examples=1000,accuracy=1.,trace_accuracy=1.,format_accuracy=1.)
    assert not failures({'criteria':{'a':perfect}},strict=True)
    for key in ('accuracy','trace_accuracy','format_accuracy'):
        bad={**perfect,key:.999}
        assert failures({'criteria':{'a':bad}},retention=True,strict=True)==['a']
        assert failures({'criteria':{'a':{**perfect,'strata':{'case':bad}}}},strict=True)==['a']
    assert not failures({'criteria':{'a':{**perfect,'trace_accuracy':.999}}},retention=True)

def test_dashboard_matches_strict_retention_gate(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'phase':'p','strict_gate':True}))
    m=dict(examples=100,accuracy=1.,format_accuracy=1.,trace_accuracy=.99)
    (tmp_path/'p_epoch_001.json').write_text(json.dumps({'phase':'p','epoch':1,'active':{'criteria':{}},'retention':{'criteria':{'a':m}}}))
    row=snapshot(tmp_path)['criterion_details'][0]
    assert not row['passed'] and row['trace_threshold']==1. and row['answer_threshold']==1.
