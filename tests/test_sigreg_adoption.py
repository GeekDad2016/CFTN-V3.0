from scripts.adopt_sigreg_endpoint import adopted_metadata

def test_adoption_counts_only_selected_arm_and_retains_stage_policy():
    anchor={'dataset_hash':'hash','stage_index':5,'completed':['earlier'],'controller':{'mode':'normal','normal_done':5,'normal_total':5,'recovery_total':57},'round':63,'cursor':1}
    spec={'rounds':10,'dataset_sha256':'hash','stage_index':5,'seed_start_round':63,'checkpoint_sha256':'start','coefficient':.0001}
    report={'active':{'criteria':{}},'retention':{'criteria':{}}};policy={'normal_rounds':240,'sigreg_coefficient':.0001}
    meta=adopted_metadata(anchor,spec,report,policy)
    assert meta['round']==73 and meta['cursor']==0
    assert meta['controller']['normal_done']==15 and meta['controller']['recovery_total']==57
    assert meta['completed']==['earlier'] and meta['policy']==policy and not meta['accepted']
    assert anchor['controller']['normal_done']==5
