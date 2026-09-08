import torch
from cftn_v3.sigreg_experiment import sigreg,regularized_loss
from cftn_v3.local_math_training import batch_loss
from cftn_v3.local_specialist import LocalMathTower
from cftn_v3.full_curriculum_data import make

def test_sigreg_finite_gradient_and_dedicated_rng():
    torch.manual_seed(10);z=torch.randn(32,16,requires_grad=True);rng=torch.get_rng_state().clone()
    loss=sigreg(z,42);loss.backward()
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    assert torch.equal(loss,sigreg(z,42))

def test_gaussian_sample_scores_better_than_collapsed_features():
    torch.manual_seed(8)
    assert sigreg(torch.randn(512,32),1)<sigreg(torch.zeros(512,32),1)

def test_zero_coefficient_preserves_cross_entropy_and_hook_is_removed():
    model=LocalMathTower({'hidden_size':8,'layers':1,'attention_heads':2,'feed_forward_size':16,'dropout':0.,'max_sequence_length':256})
    rows=[make({'op':'add','left':a,'right':2},0,'add') for a in range(3)]
    model.train();expected=batch_loss(model,rows)
    total,ce,reg=regularized_loss(model,rows,12,0.)
    assert torch.equal(expected,total) and torch.equal(expected,ce) and reg==0
    total,ce,reg=regularized_loss(model,rows,12,1e-4)
    total.backward()
    assert torch.isfinite(total) and not model.final_norm._forward_hooks
    assert torch.isfinite(model.token_embedding.weight.grad).all()
