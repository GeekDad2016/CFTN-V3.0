"""Seeded fresh V3.1 maths model; never copies V3.0 weights."""
import json,torch
from pathlib import Path
from cftn_v3.local_specialist import LocalMathTower,save_specialist
root=Path('G:/ctfn-text/artifacts/v3_1');root.mkdir(parents=True,exist_ok=True)
spec={'layers':8,'hidden_size':288,'attention_heads':8,'feed_forward_size':1232,'dropout':.1,'max_sequence_length':2048,'receiver_layers':[2,5],'answer_min':-512,'answer_max':512}
old=LocalMathTower({**spec,'hidden_size':256,'feed_forward_size':1024})
torch.manual_seed(3109307);model=LocalMathTower(spec)
with torch.no_grad():
    for name,p in model.named_parameters():
        if p.ndim>1:torch.nn.init.normal_(p,mean=0.,std=.02)
        elif name.endswith('bias'):p.zero_()
old_count=sum(p.numel() for p in old.parameters());count=sum(p.numel() for p in model.parameters())
assert .295<(count/old_count-1)<.305
path=root/'initial_math.specialist';assert not path.exists()
save_specialist(path,model,{'tower':'math','revision':'V3.1','initialization':'fresh_normal_0.02','seed':3109307})
report={'revision':'V3.1','parameters':count,'previous_parameters':old_count,'increase_percent':100*(count/old_count-1),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'spec':spec,'checkpoint':str(path)}
(root/'model_spec.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
oldconfig=json.loads(Path('config/local_curriculum_v4.json').read_text());cfg={'root':str(root),'queue':oldconfig['queue']}
for item in cfg['queue']:
    item['output']=str(root/item['tower']);item.pop('inherit_progress',None);item.pop('upgrade_validation_warmup',None)
    if item['tower']=='math':item.update(data='G:/ctfn-text/data/v3_1_main',initial_checkpoint=str(path))
Path('config/local_curriculum_v31.json').write_text(json.dumps(cfg,indent=2)+'\n')
