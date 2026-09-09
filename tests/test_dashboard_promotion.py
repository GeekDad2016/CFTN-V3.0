import json
from cftn_v3.local_math_dashboard import snapshot

def test_full_failure_overrides_perfect_routine_and_exposes_samples(tmp_path):
    metric={'examples':32,'accuracy':1.,'trace_accuracy':1.,'format_accuracy':1.}
    routine={'phase':'sums','epoch':46,'passed':True,'active':{'criteria':{'sums':metric},**metric},'retention':{}}
    failure={'prompt':'compare','expected':'>','output':'=','criterion':'sums','answer_correct':False,'trace_correct':False}
    full_metric={**metric,'examples':619,'accuracy':.955,'strata':{'>':{**metric,'accuracy':.932}}}
    promotion={'epoch':46,'passed':False,'active':{**full_metric,'criteria':{'sums':full_metric},'samples':[failure]},'retention':{}}
    for name,value in [('status.json',{'phase':'sums','state':'blocked'}),('sums_epoch_046.json',routine),('sums_promotion_validation.json',promotion)]:
        (tmp_path/name).write_text(json.dumps(value))
    data=snapshot(tmp_path)
    assert data['display_evaluation']['kind']=='Full promotion check'
    assert data['display_evaluation']['active']['accuracy']==.955
    assert not data['display_evaluation']['passed']
    assert not data['criterion_details'][0]['passed']
    assert data['criterion_details'][0]['samples']==[failure]
    assert data['samples']==[failure]
    # A newer perfect routine check still cannot supersede a failed full gate.
    (tmp_path/'sums_epoch_047.json').write_text(json.dumps({**routine,'epoch':47}))
    assert not snapshot(tmp_path)['display_evaluation']['passed']
    # A later passing full gate can supersede the failed gate.
    (tmp_path/'sums_promotion_validation.json').write_text(json.dumps({**promotion,'epoch':47,'passed':True,'active':routine['active']}))
    assert snapshot(tmp_path)['display_evaluation']['passed']
    # Results from the preceding stage must never appear on a new stage.
    (tmp_path/'status.json').write_text(json.dumps({'phase':'next','state':'starting'}))
    assert not snapshot(tmp_path)['criterion_details']

def test_legacy_promotion_report_infers_round_and_uses_gate_retention_threshold(tmp_path):
    metric={'examples':20,'accuracy':.95,'format_accuracy':1.,'trace_accuracy':1.}
    (tmp_path/'status.json').write_text(json.dumps({'phase':'sums','state':'blocked'}))
    (tmp_path/'sums_before.json').write_text(json.dumps({'retention':{'criteria':{'prior':{**metric,'accuracy':1.}}}}))
    (tmp_path/'sums_epoch_046.json').write_text(json.dumps({'phase':'sums','epoch':46,'active':{},'retention':{}}))
    (tmp_path/'sums_promotion_validation.json').write_text(json.dumps({'passed':True,'active':{},'retention':{'criteria':{'prior':metric}}}))
    data=snapshot(tmp_path)
    assert data['display_evaluation']['epoch']==46
    assert data['criterion_details'][0]['passed']


def test_manual_panel_is_labelled_and_pending_is_visible(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'phase':'sums'}))
    (tmp_path/'sums_promotion_validation.json').write_text(json.dumps({'epoch':1,'passed':False}))
    (tmp_path/'sums_manual_validation.json').write_text(json.dumps({'epoch':2,'passed':True,'active':{},'retention':{}}))
    (tmp_path/'VALIDATE_REQUEST.json').write_text('{}')
    data=snapshot(tmp_path)
    assert data['manual_validation_pending']
    assert data['display_evaluation']['kind']=='Manual validation (routine panel)'
    assert data['display_evaluation']['epoch']==2
