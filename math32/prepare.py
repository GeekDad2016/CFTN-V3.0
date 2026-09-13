"""Exhaustive tokenizer and target audit before the dataset can be trained."""
import hashlib,json
from pathlib import Path
from .data import ROOT,answer
from .model import Tokenizer
from cftn_v3.file_io import atomic_json

def run():
    groups={s:[json.loads(l) for l in (ROOT/(s+'.jsonl')).read_text().splitlines()] for s in ('train','validation','test')}
    allrows=sum(groups.values(),[]);tok=Tokenizer.create(allrows);maximum=0
    for r in allrows:
        assert answer(r['ir'])==r['answer']
        assert tok.decode(tok.encode(r['answer']))==r['answer']
        ids=tok.prefix(r['ir'])+tok.encode(r['answer'])+[2]
        maximum=max(maximum,len(ids));assert len(ids)<=256,(r['criterion'],len(ids))
        roles,positions,op=tok.features(ids,64);assert max(positions)<128
    atomic_json(ROOT/'vocab.json',tok.vocab)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    manifest['files']['vocab.json']=hashlib.sha256((ROOT/'vocab.json').read_bytes()).hexdigest()
    manifest['tokenizer']={'vocab_size':512,'active_tokens':sum(not v.startswith('<RESERVED_') for v in tok.vocab),
       'digit_order':'least significant first','max_encoded_length':maximum,'role_layout':'argument fields then result','position_offset_train':[1,64]}
    atomic_json(ROOT/'manifest.json',manifest)
    print(json.dumps({'records_checked':len(allrows),**manifest['tokenizer']}))

if __name__=='__main__':run()
