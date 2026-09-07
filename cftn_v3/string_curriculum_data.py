"""Prepare the next exact ASCII specialist with source-disjoint held-out strings."""
import argparse
import collections
import json
from pathlib import Path
from .config import canonical,identity
from .data import file_hash
from .full_curriculum_data import verify_manifest
from .local_specialist import MathTokenizer,load_legacy,save_specialist

OPS=['length','count','index','reverse','contains','substitute']

def convert(raw):
    op=raw['operation'];s=raw['source_string'];ir={'op':op,'text':s}
    if not s.isascii():raise ValueError('ASCII contract violation')
    if op=='length':value=str(len(s))
    elif op=='count':ir['character']=raw['character'];value=str(s.count(ir['character']))
    elif op=='index':ir['index']=raw['index'];value=s[ir['index']]
    elif op=='reverse':value=s[::-1]
    elif op=='contains':ir['substring']=raw['substring'];value='yes' if ir['substring'] in s else 'no'
    elif op=='substitute':ir.update(character=raw['character'],replacement=raw['replacement']);value=s.replace(ir['character'],ir['replacement'])
    else:raise ValueError('Unknown string operation')
    if value!=raw['value'] or raw['target_answer']!=f'<answer>{value}</answer>':raise ValueError('String answer mismatch')
    # Retain the checked legacy trace and use a minimal typed request.
    return {'ir':ir,'prompt':canonical(ir),'target':raw['target_trace'],'answer':value,'stage':OPS.index(op),
        'phase':str(OPS.index(op)),'criterion':op,'semantic_id':identity(ir),'source_record':raw['record_id']}

def build(source,destination,checkpoint):
    source,destination=Path(source),Path(destination);destination.mkdir(parents=True,exist_ok=True)
    if (destination/'manifest.json').exists():return verify_manifest(destination)
    old=json.loads((source/'manifest.json').read_text());groups={};seen_strings=set();files={};excluded={};maximum=0;tok=MathTokenizer()
    for split in ('train','validation','test'):
        meta=old['splits']['string_'+split];path=source/meta['path']
        if file_hash(path)!=meta['sha256']:raise ValueError('String source checksum mismatch')
        rows={};excluded[split]=0;strings=set()
        for line in path.open(encoding='utf-8'):
            raw=json.loads(line);r=convert(raw);s=raw['source_string']
            if s in seen_strings:excluded[split]+=1;continue
            strings.add(s);rows[r['semantic_id']]=r
            maximum=max(maximum,len(tok.prefix(r['prompt']))+len(tok.encode(r['target']))+1)
        seen_strings.update(strings);groups[split]=list(rows.values());p=destination/(split+'.jsonl')
        p.write_text(''.join(canonical(r)+'\n' for r in rows.values()),encoding='utf-8');files[p.name]=file_hash(p)
    if maximum>256:raise ValueError('String context overflow: '+str(maximum))
    stages=[]
    for i,op in enumerate(OPS):
        rows=sorted([r for r in groups['train'] if r['stage']==i],key=lambda r:len(r['ir']['text']))[:512]
        path=destination/f'remediation_{i:02d}.jsonl';path.write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8');files[path.name]=file_hash(path)
        stages.append({'index':i,'name':op,'scope':'Exact ASCII '+op,'criteria':[op],'remediation':path.name,
            'train_records':sum(r['stage']==i for r in groups['train']),'validation_records':sum(r['stage']==i for r in groups['validation'])})
    spec={'hidden_size':256,'layers':6,'attention_heads':8,'feed_forward_size':1024,'dropout':.1,
        'max_sequence_length':256,'receiver_layers':[2,5],'answer_min':-512,'answer_max':512}
    model=load_legacy(checkpoint,spec)
    save_specialist(destination/'initial.specialist',model,{'tower':'string','accepted':False,'source_checkpoint':str(checkpoint)})
    files['initial.specialist']=file_hash(destination/'initial.specialist')
    result={'format':'full_string_curriculum_v2','stages':stages,'files':files,'records':{k:len(v) for k,v in groups.items()},
        'excluded_source_string_overlap':excluded,'max_training_tokens':maximum,'source':str(source),
        'scope':'Exact ASCII strings: length, count, zero-based index, reverse, contains, substitution. No Unicode or general language claim.'}
    (destination/'manifest.json').write_text(canonical(result),encoding='utf-8');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args()
    r=build(a.source,a.output,a.checkpoint);print({k:v for k,v in r.items() if k not in ('files','stages')})
