import pytest
import torch
from cftn_v3.learning_experiment import record,authorized_record
from cftn_v3.config import Config
from cftn_v3.model import CFTN,ByteTokenizer
from cftn_v3.training import make_plan,train
from cftn_v3.data import example

def test_experiment_opt_in_and_cuda():
    r=record('math','What is 2+2?','5','experiment')
    r['teacher_revision']=Config().revision
    with pytest.raises(ValueError):make_plan('continual',('math',),[r])
    plan=make_plan('continual',('math',),[r],verifier=authorized_record)
    model=CFTN(Config.tiny(),ByteTokenizer()).to('cuda' if torch.cuda.is_available() else 'cpu')
    state=train(model,[r],plan,1,replay=[example('math',0,'en')],verifier=authorized_record)
    assert state['frozen_hashes_verified']
    assert all(k.startswith('towers.math.') for k in state['changed'])
