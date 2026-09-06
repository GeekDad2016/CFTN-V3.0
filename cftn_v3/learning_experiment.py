"""Automatic multi-domain learning experiment; never publishes accepted weights."""
import argparse
import itertools
import gc
import json
import time
from pathlib import Path
import torch
from .config import Config,canonical,identity
from .data import read_rows,example,composition,file_hash
from .answer_contracts import CONTRACT_VERSION, FORMATS, canonical_answer, correct, validated_target

BLOCK_STEPS = 1000
CHECKPOINT_STEPS = 200
FRESH_EXAMPLES = 1024

DOMAINS=('math','retrieval','code','formal_logic')
BOOLQ_REV='35b264d03638db9f4ce671b711558bf7ff0f80d5'


def record(tower,prompt,target,source):
    return dict(id=identity([source,prompt]),semantic_id=identity(prompt),tower=tower,language='en',
        prompt=prompt,target=target,reference=target,source=source,criterion=source,
        verifier='experimental_teacher_not_truth_verified',verified=False,
        routing={'targets':[tower],'rounds':{tower:0}})


def prepare(root):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from .teacher_cycles import prepare as gsm_prepare
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        manifest=json.loads((root/'manifest.json').read_text())
        for name,digest in manifest['files'].items():
            if file_hash(root/name)!=digest:raise ValueError('experiment data changed')
        return
    gsm_prepare(root/'gsm8k')
    groups={t:[] for t in DOMAINS}
    for r in read_rows(root/'gsm8k/train.jsonl'):
        groups['math'].append(record('math',r['prompt'],r['target'],'gsm8k'))
    path=hf_hub_download('google/boolq','data/train-00000-of-00001.parquet',repo_type='dataset',revision=BOOLQ_REV,local_dir=root/'boolq')
    for r in pq.read_table(path).to_pylist():
        if len(r['passage'])>1800:continue
        groups['retrieval'].append(record('retrieval',r['passage']+'\nQuestion: '+r['question']+'\nAnswer only yes or no.',
            'yes' if r['answer'] else 'no','boolq'))
    for i in range(512):
        a=2+i%97;b=3+i//97
        groups['code'].append(record('code',f'Write only a Python function solve(x) that returns x * {a} + {b}.',
            f'def solve(x):\n    return x * {a} + {b}','synthetic_python'))
        groups['formal_logic'].append(record('formal_logic',f'Given P({i}), P(x) implies Q(x), Q(x) implies R(x). List the three propositions proving R({i}), separated by semicolons.',
            f'P({i});Q({i});R({i})','synthetic_logic'))
    files={}
    for tower,items in groups.items():
        items=list({r['semantic_id']:r for r in items}.values())
        items.sort(key=lambda r:identity(r['id']))
        for split,subset in [('heldout',items[:32]),('train',items[32:])]:
            out=root/f'{tower}_{split}.jsonl';out.write_text(''.join(canonical(r)+'\n' for r in subset),encoding='utf-8');files[out.name]=file_hash(out)
    (root/'manifest.json').write_text(canonical({'files':files,'sources':{
        'gsm8k':{'repo':'openai/gsm8k','revision':'740312add88f781978c0658806c59bc2815b9866','license':'MIT'},
        'boolq':{'repo':'google/boolq','revision':BOOLQ_REV,'license':'CC-BY-SA-3.0'},
        'synthetic':'local bounded Python, logic and Math-to-String tasks'},
        'scope':'Learning experiment only. Teacher answers can be wrong. Held-out questions never enter training.'}))


def authorized_record(r):
    # Explicit experimental caller only. Default verification/ingestion is unchanged.
    return r.get('tower') in DOMAINS and bool(r.get('prompt')) and bool(r.get('target')) and r.get('teacher_revision')==Config().revision


def teach(rows,path,writer):
    if path.exists():
        cached=read_rows(path)
        if all(r.get('contract_version')==CONTRACT_VERSION for r in cached):return cached
    from transformers import AutoTokenizer,AutoModelForCausalLM
    config=Config()
    tokenizer=AutoTokenizer.from_pretrained(config.coordinator,revision=config.revision)
    model=AutoModelForCausalLM.from_pretrained(config.coordinator,revision=config.revision,torch_dtype=torch.bfloat16).to('cuda').eval()
    output=[]
    try:
        tokenizer.padding_side='left'
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        for i in range(0,len(rows),8):
            batch=rows[i:i+8]
            prompts=[tokenizer.apply_chat_template([
                {'role':'system','content':'Follow the output contract exactly. '+FORMATS[r['tower']]},
                {'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True) for r in batch]
            ids=tokenizer(prompts,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
            with torch.no_grad():answer=model.generate(**ids,max_new_tokens=256,do_sample=False,pad_token_id=tokenizer.pad_token_id)
            eos=model.generation_config.eos_token_id
            eos=set(eos if isinstance(eos,list) else [eos])
            for r,tokens in zip(batch,answer[:,ids['input_ids'].shape[1]:]):
                text=tokenizer.decode(tokens,skip_special_tokens=True).strip()
                row=validated_target(r,text,complete=any(t in eos for t in tokens.tolist()))
                output.append({**row,'teacher_revision':config.revision,'verified':False})
            writer({'phase':'teacher','state':'generating','targets':[batch[0]['tower']],
                    'completed':len(output),'total':len(rows),
                    'teacher_accepted':sum(r['teacher_accepted'] for r in output)})
    finally:
        del model
        gc.collect();torch.cuda.empty_cache()
    tmp=path.with_suffix('.tmp');tmp.write_text(''.join(canonical(r)+'\n' for r in output),encoding='utf-8');tmp.replace(path)
    return output


def measure(model,rows,tower,sample_count=16):
    from .training import supervised_loss
    values=[];samples=[]
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for r in rows:
            reference={**r,'target':r.get('reference',r['target'])}
            values.append(float(supervised_loss(model,reference,tower)))
        for r in rows[:sample_count]:
            answer=model.generate(r['prompt']+'\n',tower,max_tokens=128)
            samples.append({'prompt':r['prompt'],'expected':r.get('reference',r['target']),'output':answer,
                'exact_reference_match':answer.strip()==r.get('reference',r['target']).strip(),
                'correct':correct(tower,answer,r.get('reference',r['target'])) or answer.strip()==r.get('reference',r['target']).strip()})
    return {'reference_loss':sum(values)/len(values),'exact_reference_matches':sum(r['exact_reference_match'] for r in samples),
        'correct_answers':sum(r['correct'] for r in samples),
        'accuracy':sum(r['correct'] for r in samples)/max(1,len(samples)),
        'sample_count':len(samples),'samples':samples,'note':'Fixed held-out panel; bounded numeric and Python polynomial equivalence checks.'}



def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--wait',action='store_true');a=p.parse_args()
    prepare('data/learning_experiment')
    from .learning_cycle_v2 import prepare_v2,run as run_v2
    prepare_v2('data/learning_experiment','data/learning_experiment_v2')
    if a.prepare_only:return
    if a.wait:
        while not Path('artifacts/tower_repairs/final.json').exists() or Path('artifacts/gpu.lock').exists():time.sleep(30)
    if not Path('artifacts/tower_repairs/final.json').exists():raise RuntimeError('Wait for repairs to finish')
    try:
        run_v2('artifacts/learning_experiment','data/learning_experiment_v2')
    except Exception as exc:
        from .runtime import status_writer
        status_writer('artifacts')({'state':'failed','phase':'experiment','error':str(exc)})
        raise


if __name__=='__main__':main()
