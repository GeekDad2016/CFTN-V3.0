"""Digit-preserving grammar, role and randomized Abacus positions, causal transformer."""
import json, math, re, string
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

class Tokenizer:
    def __init__(self,vocab):
        self.vocab=vocab;self.ids={s:i for i,s in enumerate(vocab)}
        self.digits=set(self.ids[str(i)] for i in range(10));self.keys={i for s,i in self.ids.items() if s.startswith('<K:')}
        self.ops={s[4:-1]:i for s,i in self.ids.items() if s.startswith('<OP:')}
    @classmethod
    def create(cls,records):
        base=['<PAD>','<BOS>','<EOS>','<SEP>','<RESULT>','<STEP>']+list('0123456789')
        base += ['<OP:'+op+'>' for op in sorted({r['ir']['op'] for r in records})]
        base += ['<K:'+k+'>' for k in sorted({k for r in records for k in r['ir'] if k!='op'})]
        base += [c for c in string.printable if c not in base and c not in '\r\n\t\x0b\x0c']
        assert len(base)<384
        return cls(base+[f'<RESERVED_{i}>' for i in range(512-len(base))])
    def encode(self,text):
        text=re.sub(r'\d+',lambda m:m[0][::-1],str(text))
        return [self.ids[c] for c in text]
    def decode(self,tokens):
        text=''.join(self.vocab[int(t)] for t in tokens if int(t) not in (0,2))
        return re.sub(r'\d+',lambda m:m[0][::-1],text)
    def prefix(self,ir):
        ids=[1,self.ops[ir['op']]]
        for k,v in sorted(ir.items()):
            if k=='op':continue
            ids += [self.ids['<K:'+k+'>']]+self.encode(json.dumps(v,separators=(',',':')))+[3]
        return ids+[4]
    def features(self,ids,offset=0):
        # Every role and numerical position depends only on tokens already visible.
        role=0;position=0;roles=[];positions=[];operation=0
        for t in ids:
            if t in self.ops.values():operation=t
            if t in self.keys:role=min(role+1,12)
            if t==4:role=13
            if t==5:role=14
            position=position+1 if t in self.digits else 0
            roles.append(role);positions.append(position+offset if position else 0)
        return roles,positions,operation

class Norm(nn.Module):
    def __init__(self,d):super().__init__();self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x):return x*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6).to(x.dtype)*self.weight

class Block(nn.Module):
    def __init__(self,d,heads,ff):
        super().__init__();self.heads=heads;self.n1=Norm(d);self.n2=Norm(d)
        self.qkv=nn.Linear(d,3*d,bias=False);self.out=nn.Linear(d,d,bias=False)
        self.gate=nn.Linear(d,ff,bias=False);self.up=nn.Linear(d,ff,bias=False);self.down=nn.Linear(ff,d,bias=False)
    def forward(self,x):
        b,n,d=x.shape;q,k,v=self.qkv(self.n1(x)).chunk(3,-1)
        q,k,v=[t.view(b,n,self.heads,d//self.heads).transpose(1,2) for t in (q,k,v)]
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,n,d)
        x=x+self.out(a);y=self.n2(x);return x+self.down(F.silu(self.gate(y))*self.up(y))

class Tower(nn.Module):
    def __init__(self,tokenizer,d=384,layers=6,heads=6,ff=1536,context=256):
        super().__init__();self.tok=tokenizer;self.context=context
        self.spec=dict(d=d,layers=layers,heads=heads,ff=ff,context=context)
        self.token=nn.Embedding(512,d);self.column=nn.Embedding(128,d);self.role=nn.Embedding(16,d);self.operation=nn.Embedding(512,d)
        self.blocks=nn.ModuleList(Block(d,heads,ff) for _ in range(layers));self.norm=Norm(d)
        self.head=nn.Linear(d,512,bias=False);self.head.weight=self.token.weight
        pos=torch.arange(context)[:,None];freq=torch.exp(torch.arange(0,d,2)*(-math.log(10000)/d))
        seq=torch.zeros(context,d);seq[:,0::2]=torch.sin(pos*freq);seq[:,1::2]=torch.cos(pos*freq)
        self.register_buffer('sequence',seq)
        self.apply(self.initialize)
    @staticmethod
    def initialize(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):nn.init.normal_(m.weight,std=.02)
    def forward(self,ids,roles,positions,ops,return_hidden=False):
        assert ids.shape[1]<=self.context and int(positions.max())<128
        x=self.token(ids)+self.column(positions)+self.role(roles)+self.operation(ops)[:,None]+self.sequence[:ids.shape[1]]
        for b in self.blocks:x=b(x)
        h=self.norm(x);logits=self.head(h)
        # Reserved IDs are inactive. They can be assigned in a future version without renumbering.
        inactive=[i for i,s in enumerate(self.tok.vocab) if s.startswith('<RESERVED_')]
        logits[:,:,inactive]=float('-inf')
        return (logits,h) if return_hidden else logits
    @torch.inference_mode()
    def generate(self,ir,max_tokens=128):
        self.eval();ids=self.tok.prefix(ir);out=[];device=self.token.weight.device
        for _ in range(min(max_tokens,self.context-len(ids))):
            r,p,o=self.tok.features(ids)
            args=[torch.tensor([v],device=device) for v in (ids,r,p)]+[torch.tensor([o],device=device)]
            token=int(self(*args)[0,-1].argmax())
            if token==2:return self.tok.decode(out),True
            ids.append(token);out.append(token)
        return self.tok.decode(out),False

def batch(tokenizer,rows,device,offsets=None):
    allids=[];allroles=[];allpos=[];ops=[];labels=[];ends=[]
    for i,row in enumerate(rows):
        prefix=tokenizer.prefix(row['ir']);ids=prefix+tokenizer.encode(row['answer'])+[2]
        r,p,o=tokenizer.features(ids,offsets[i] if offsets else 0)
        assert len(ids)<=256,('Context overflow',row['criterion'],len(ids))
        allids.append(ids);allroles.append(r);allpos.append(p);ops.append(o);ends.append(len(prefix)-1)
        labels.append([-100]*len(prefix)+ids[len(prefix):])
    length=max(map(len,allids))
    def tensor(xs,pad):return torch.tensor([x+[pad]*(length-len(x)) for x in xs],device=device)
    return tensor(allids,0),tensor(allroles,0),tensor(allpos,0),torch.tensor(ops,device=device),tensor(labels,-100),ends
