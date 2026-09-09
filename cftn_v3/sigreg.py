"""Prompt-feature SIGReg used by the paired trial and curriculum."""
import torch
from .local_math_training import batch_loss
from .local_specialist import MathTokenizer

def sigreg(features,seed,projections=256):
    # A local generator does not disturb the shared dropout RNG sequence.
    with torch.autocast(features.device.type,enabled=False):
        z=features.float();g=torch.Generator(device=z.device);g.manual_seed(seed)
        directions=torch.randn(z.shape[-1],projections,device=z.device,generator=g)
        directions=directions/directions.norm(dim=0,keepdim=True).clamp_min(1e-8)
        t=torch.linspace(0,3,17,device=z.device);phi=torch.exp(-t.square()/2)
        weights=torch.full_like(t,6/16);weights[0]=weights[-1]=3/16
        angles=(z@directions).unsqueeze(-1)*t
        discrepancy=(angles.cos().mean(0)-phi).square()+angles.sin().mean(0).square()
        return ((discrepancy*(weights*phi)).sum(-1)*len(z)).mean()

def regularized_loss(model,rows,seed,coefficient):
    captured=[]
    def capture(module,args,output):
        positions=torch.tensor([len(MathTokenizer().prefix(r['prompt']))-1 for r in rows],device=output.device)
        captured.append(output[torch.arange(len(rows),device=output.device),positions])
    handle=model.final_norm.register_forward_hook(capture) if coefficient else None
    try:ce=batch_loss(model,rows)
    finally:
        if handle:handle.remove()
    penalty=sigreg(captured[0],seed) if coefficient else ce.new_zeros(())
    return ce+coefficient*penalty,ce,penalty

