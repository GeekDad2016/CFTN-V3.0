import torch
from math32.model import Tokenizer,Tower,batch
from math32.data import answer,carries,family
from math32.train import sample

def fixture():
    row={'ir':{'op':'add','left':37,'right':18},'answer':'55','criterion':'add','stage':0,'case':'carry'}
    return row,Tokenizer.create([row])

def test_digits_roles_and_loss_mask():
    row,tok=fixture()
    assert tok.encode('123456')==[tok.ids[c] for c in '654321']
    assert tok.decode(tok.encode('-123/45'))=='-123/45'
    ids,roles,pos,ops,labels,ends=batch(tok,[row],'cpu')
    assert (labels[0,:ends[0]+1]==-100).all()
    assert labels[0,ends[0]+1:].tolist()==tok.encode('55')+[2]
    prefix=tok.prefix(row['ir']);r,p,o=tok.features(prefix)
    r2,p2,o2=tok.features(prefix+tok.encode('55'))
    assert r2[:len(r)]==r and p2[:len(p)]==p and o2==o

def test_causal_forward_and_cuda_update():
    row,tok=fixture();device='cuda' if torch.cuda.is_available() else 'cpu'
    model=Tower(tok,d=48,layers=2,heads=6,ff=96).to(device)
    ids,r,p,o,y,_=batch(tok,[row],device)
    logits=model(ids,r,p,o)
    changed=ids.clone();changed[:,-1]=tok.ids['8']
    assert torch.allclose(logits[:,:-1],model(changed,r,p,o)[:,:-1],atol=1e-5,equal_nan=True)
    loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,512),y[:,1:].reshape(-1),ignore_index=-100)
    assert torch.isfinite(loss);loss.backward()
    assert model.blocks[0].qkv.weight.grad.abs().sum()>0

def test_regrouping_and_balanced_replay():
    assert answer({'op':'regroup','value':18})=='[0,18]'
    assert answer({'op':'column_subtract','left':18,'right':9})=='[9,0,9]'
    assert carries(999,1)==(3,3)
    assert family({'op':'add','left':1,'right':4})==family({'op':'add','left':4,'right':1})
    rows=[{'stage':0,'case':'a'},{'stage':0,'case':'b'}]*20
    chosen=sample(rows,100,1)
    assert sum(r['case']=='a' for r in chosen)==50
