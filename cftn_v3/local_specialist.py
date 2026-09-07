"""Portable V12-compatible Maths specialist; no coordinator is loaded locally."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .file_io import atomic_write


class MathTokenizer:
    vocab_size=260
    pad_token_id=0
    eos_token_id=2

    def encode(self,text,**kwargs):
        return [v+4 for v in text.encode('utf-8')]

    def decode(self,ids,**kwargs):
        return bytes(int(v)-4 for v in ids if 4<=int(v)<260).decode('utf-8',errors='replace')

    def prefix(self,problem):
        return [1]+self.encode('Problem: '+problem.rstrip('\n')+'\nSolution:')+[3]


class LocalMathTower(nn.Module):
    """Matches legacy parameter names exactly; retains independent byte vocabulary."""
    def __init__(self,spec):
        super().__init__();self.spec=dict(spec)
        self.width=int(spec['hidden_size']);self.context=int(spec['max_sequence_length'])
        self.token_embedding=nn.Embedding(260,self.width)
        self.position_embedding=nn.Embedding(self.context,self.width)
        self.dropout=nn.Dropout(spec.get('dropout',.1))
        self.blocks=nn.ModuleList(nn.TransformerEncoderLayer(self.width,spec['attention_heads'],
            spec['feed_forward_size'],spec.get('dropout',.1),activation='gelu',batch_first=True,norm_first=True)
            for _ in range(spec['layers']))
        self.final_norm=nn.LayerNorm(self.width)
        self.lm_head=nn.Linear(self.width,260,bias=False);self.lm_head.weight=self.token_embedding.weight
        self.answer_head=nn.Linear(self.width,spec.get('answer_max',512)-spec.get('answer_min',-512)+1)
        # Legacy auxiliary classifier is preserved for checkpoint compatibility only.
        self.answer_head.requires_grad_(False)

    def hidden(self,ids,message=None,receiver=None):
        if ids.shape[1]>self.context:raise ValueError('Math context overflow')
        pos=torch.arange(ids.shape[1],device=ids.device)[None]
        x=self.dropout(self.token_embedding(ids)+self.position_embedding(pos))
        mask=torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool).triu(1)
        for i,block in enumerate(self.blocks):
            x=block(x,src_mask=mask,src_key_padding_mask=ids.eq(0))
            if message is not None and receiver is not None and i in self.spec.get('receiver_layers',[]):
                x=receiver(x,message)
        return self.final_norm(x)

    def forward(self,ids):return self.lm_head(self.hidden(ids))

    @torch.no_grad()
    def generate(self,problem,max_tokens=384):
        """Incremental KV decode using the existing encoder-block weights."""
        self.eval();tok=MathTokenizer();device=next(self.parameters()).device
        ids=torch.tensor([tok.prefix(problem)],device=device);output=[];cache=[]
        if ids.shape[1]>=self.context:raise ValueError('Math prefix exceeds context')
        length=0
        for step in range(min(max_tokens,self.context-ids.shape[1])):
            x=self.token_embedding(ids)+self.position_embedding(torch.arange(length,length+ids.shape[1],device=device)[None])
            for i,block in enumerate(self.blocks):
                q,k,v=F.linear(block.norm1(x),block.self_attn.in_proj_weight,block.self_attn.in_proj_bias).chunk(3,-1)
                heads=block.self_attn.num_heads;dim=self.width//heads
                q,k,v=[t.view(1,-1,heads,dim).transpose(1,2) for t in (q,k,v)]
                if step:k=torch.cat((cache[i][0],k),2);v=torch.cat((cache[i][1],v),2);cache[i]=(k,v)
                else:cache.append((k,v))
                a=F.scaled_dot_product_attention(q,k,v,is_causal=not step).transpose(1,2).reshape(1,-1,self.width)
                x=x+block.self_attn.out_proj(a)
                x=x+block.linear2(block.activation(block.linear1(block.norm2(x))))
            token=int(self.lm_head(self.final_norm(x[:,-1:])).argmax(-1))
            length+=ids.shape[1]
            if token==tok.eos_token_id:return tok.decode(output),True
            output.append(token);ids=torch.tensor([[token]],device=device)
        return tok.decode(output),False


def load_legacy(checkpoint,spec):
    # Explicitly selected, trusted local experiment checkpoint.
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=LocalMathTower(spec)
    model.load_state_dict(saved['model_state'],strict=True)
    return model


def save_specialist(path,model,metadata,optimizer=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tower=metadata.get('tower','math')
    payload={'format':'cftn_native_specialist_v1','tower':tower,'tokenizer':'legacy_math_bytes_260',
        'interface':'typed_string_ir_v1' if tower=='string' else 'typed_math_ir_v1','spec':model.spec,
        'weights':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        'metadata':metadata,'optimizer':optimizer.state_dict() if optimizer else None,
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    atomic_write(path,lambda stream:torch.save(payload,stream))


def load_specialist(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['format']!='cftn_native_specialist_v1' or saved['tokenizer']!='legacy_math_bytes_260':
        raise ValueError('Unsupported specialist format')
    model=LocalMathTower(saved['spec']);model.load_state_dict(saved['weights'],strict=True)
    return model,saved


def install_specialist(model,path,allow_experimental=False):
    specialist,saved=load_specialist(path)
    if not allow_experimental and not saved['metadata'].get('accepted'):
        raise ValueError('Specialist has not passed its acceptance checks')
    from .model import Bridge,Receiver
    from .data import file_hash
    name=saved['tower'];device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    model.towers[name]=specialist.to(device=device,dtype=dtype)
    model.bridges[name]=nn.ModuleDict({'request':Bridge(model.coordinator.width,specialist.width,model.config.message_tokens),
        'return':Bridge(specialist.width,model.coordinator.width,model.config.message_tokens),
        'receiver':Receiver(specialist.width)}).to(device=device,dtype=dtype)
    model.config.specialist_specs[name]={'kind':'legacy_string_v13' if name=='string' else 'legacy_math_v12','spec':saved['spec'],
        'sha256':file_hash(Path(path)),'interface':saved['interface'],'metadata':saved['metadata']}
    # Fresh bridges are untrained. Native readiness never activates coordinated inference.
    model.config.active=tuple(t for t in model.config.active if t!=name)


def main():
    import argparse
    from .artifact import load_bundle,save_bundle
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--specialist',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.base).resolve()==Path(a.output).resolve():raise ValueError('Assembly must preserve the base checkpoint')
    model,_,meta=load_bundle(a.base)
    install_specialist(model,a.specialist)
    save_bundle(a.output,model,metadata={**meta,'assembly_requires_coordination_training':True,'accepted_release':False})


if __name__=='__main__':main()
