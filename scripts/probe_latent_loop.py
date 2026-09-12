"""Read-only failed-question diagnostic: one vs two shared-weight latent passes."""
import json, time, shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from cftn_v3.local_specialist import load_specialist, MathTokenizer
from cftn_v3.math_procedures import score
from cftn_v3.full_curriculum_data import make
from cftn_v3.data import file_hash
from cftn_v3.file_io import atomic_json

@torch.inference_mode()
def generate(model, problem, passes, max_tokens=384):
    model.eval(); tok=MathTokenizer(); device=next(model.parameters()).device
    ids=torch.tensor([tok.prefix(problem)],device=device);output=[];cache=[];length=0
    for step in range(min(max_tokens,model.context-ids.shape[1])):
        x=model.token_embedding(ids)+model.position_embedding(torch.arange(length,length+ids.shape[1],device=device)[None])
        for repeat in range(passes):
            for i,block in enumerate(model.blocks):
                ci=repeat*len(model.blocks)+i
                q,k,v=F.linear(block.norm1(x),block.self_attn.in_proj_weight,block.self_attn.in_proj_bias).chunk(3,-1)
                heads=block.self_attn.num_heads;dim=model.width//heads
                q,k,v=[t.view(1,-1,heads,dim).transpose(1,2) for t in (q,k,v)]
                if step:k=torch.cat((cache[ci][0],k),2);v=torch.cat((cache[ci][1],v),2);cache[ci]=(k,v)
                else:cache.append((k,v))
                a=F.scaled_dot_product_attention(q,k,v,is_causal=not step).transpose(1,2).reshape(1,-1,model.width)
                x=x+block.self_attn.out_proj(a)
                x=x+block.linear2(block.activation(block.linear1(block.norm2(x))))
        token=int(model.lm_head(model.final_norm(x[:,-1:])).argmax(-1))
        length+=ids.shape[1]
        if token==tok.eos_token_id:return tok.decode(output),True
        output.append(token);ids=torch.tensor([[token]],device=device)
    return tok.decode(output),False

def run():
    production=Path('G:/ctfn-text/artifacts/v3_1/math')
    report=json.loads((production/'y1_add_sub_fluency_promotion_validation.json').read_text())
    root=production.parent/('latent_loop_'+time.strftime('%Y%m%d_%H%M%S'));root.mkdir()
    # Atomic checkpoint writers replace the file: the copy sees one complete version.
    shutil.copy2(production/'current.specialist',root/'snapshot.specialist')
    digest=file_hash(root/'snapshot.specialist')
    model,saved=load_specialist(root/'snapshot.specialist');model.to('cuda');model.eval();torch.set_num_threads(2)
    selected=[]
    for split in ('active','retention'):
        for s in report[split]['samples']:
            if not all(s[k] for k in ('answer_correct','trace_correct','format_correct')):
                selected.append((split,s))
    info={'checkpoint_round':saved['metadata']['round'],'checkpoint_cursor':saved['metadata']['cursor'],
          'checkpoint_sha256':digest,'selection_report_round':report['epoch'],
          'method':'Two raw residual passes through shared blocks; embeddings once, final norm once, independent per-pass KV caches; greedy float32, max 384 tokens.',
          'selected':len(selected),'rows':[]}
    atomic_json(root/'results.json',info)
    print('Diagnostic',root,'questions',len(selected),flush=True)
    for split,s in selected:
        row=make(json.loads(s['prompt']),int(s['phase']),s['criterion'])
        assert row['target']==s['expected_trace']
        original=model.generate(row['prompt'])
        normal=generate(model,row['prompt'],1)
        assert normal==original,'One-pass implementation differs from production decoding'
        torch.cuda.synchronize();begun=time.perf_counter()
        looped=generate(model,row['prompt'],2)
        torch.cuda.synchronize()
        result={'split':split,'prompt':row['prompt'],'expected':row['answer'],'expected_trace':row['target'],
                'normal':{'output':normal[0],'complete':normal[1],**score(normal[0],row)},
                'looped':{'output':looped[0],'complete':looped[1],**score(looped[0],row)},
                'loop_seconds':time.perf_counter()-begun}
        info['rows'].append(result);atomic_json(root/'results.json',info)
        print(split,row['answer'],result['normal']['answer_correct'],result['looped']['answer_correct'],flush=True)
    info['summary']={}
    for subset in ('selected','currently_failing'):
        rows=info['rows'] if subset=='selected' else [r for r in info['rows'] if not all(r['normal'][k] for k in ('answer_correct','trace_correct','format_correct'))]
        info['summary'][subset]={'examples':len(rows),**{arm:{k:sum(bool(r[arm][k]) for r in rows)
                 for k in ('answer_correct','trace_correct','format_correct')} for arm in ('normal','looped')}}
    assert file_hash(root/'snapshot.specialist')==digest
    atomic_json(root/'results.json',info);print(json.dumps(info['summary']),flush=True)

if __name__=='__main__':run()
