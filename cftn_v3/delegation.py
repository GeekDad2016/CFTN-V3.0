"""Train and measure automatic routing, subtask formulation and synthesis.

First scope: bounded Math->String / Math->Code compositions and single calls.
No claim of general-purpose autonomous planning.
"""
import json
from .config import canonical,identity,TOWERS
from .contracts import Call,ExecutionPlan

TARGETS=('math','string','code')
SOURCE='experimental_delegation_v1'


def task(tower,prompt,target,routing=None):
    return {'id':identity([tower,prompt,target]),'tower':tower,'prompt':prompt,'target':target,
        'source':SOURCE,'language':'en','criterion':'delegation_'+tower,'verifier':'synthetic_trace_v1',
        'routing':routing or {'targets':[tower],'rounds':{tower:0},'requests':{tower:prompt}}}


def traces(start,count):
    rows=[]
    for index in range(start,start+count):
        a=2+index%97;b=2+index//97
        math=task('math',f'Calculate {a}+{b}.',f'<work>{a}+{b}={a+b}</work><answer>{a+b}</answer>')
        string=task('string',f'Reverse exactly: {a+b}',str(a+b)[::-1])
        code=task('code',f'Write Python function solve(x) that adds {a+b}.',f'def solve(x):\n    return x + {a+b}')
        rows.extend([math,string,code])
        for other,request,wording in [(string,'Reverse the digits of the result supplied by Math.','reverse the result digits'),
                                      (code,'Write only Python function solve(x) adding the result supplied by Math.','write Python solve(x) that adds the result to x')]:
            t=other['tower']
            route={'targets':['math',t],'rounds':{'math':0,t:1},'dependencies':{t:['math']},
                   'requests':{'math':math['prompt'],t:request}}
            r=task('math',f'Calculate {a}+{b}, then {wording}. Return only the final result.',other['target'],route)
            r['specialist_targets']={'math':math,t:other}
            rows.append(r)
    return rows


def verified(r):return r.get('source')==SOURCE and r.get('tower') in TOWERS and bool(r.get('target'))


def planner_prompt(prompt,plan):
    spec=[{'tower':c.tower,'round':c.round,'depends_on':list(c.depends_on)} for c in plan.calls]
    return 'Write a JSON object mapping each selected tower name to its concrete subtask request. Keep dependencies: later requests may refer to results from earlier towers. Output only JSON.\nSelected plan: '+canonical(spec)+'\nQuestion: '+prompt


def oracle(r):
    route=r['routing']
    return ExecutionPlan([Call(t,route['rounds'][t],route.get('requests',{}).get(t,r['prompt']),tuple(route.get('dependencies',{}).get(t,()))) for t in route['targets']]).validate(TARGETS)


def parse_requests(text,plan):
    if text.strip().startswith('```'):
        lines=text.strip().splitlines();text='\n'.join(lines[1:-1])
    obj=json.loads(text)
    if not isinstance(obj,dict) or set(obj)!={c.tower for c in plan.calls}:raise ValueError('planner changed selected towers')
    if any(not isinstance(v,str) or not 1<=len(v)<=1500 for v in obj.values()):raise ValueError('invalid subtask request')
    return ExecutionPlan([Call(c.tower,c.round,obj[c.tower],c.depends_on) for c in plan.calls],plan.confidence)


def automatic(model,prompt,max_tokens=128):
    try:
        plan=model.route(prompt)
        if not plan.calls:return {'answer':model.generate(prompt+'\n',max_tokens=max_tokens),'calls':[],'planner_valid':True}
        requests=model.generate(planner_prompt(prompt,plan)+'\n',max_tokens=256)
        plan=parse_requests(requests,plan).validate(model.config.active,threshold=model.config.threshold)
        return {'answer':model.generate(prompt+'\n',max_tokens=max_tokens,plan=plan),
                'calls':[{'tower':c.tower,'request':c.request,'round':c.round,'depends_on':list(c.depends_on)} for c in plan.calls],
                'planner_valid':True}
    except (ValueError,TypeError,KeyError) as exc:
        return {'answer':model.generate(prompt+'\n',max_tokens=max_tokens),'calls':[],
                'planner_valid':False,'fallback_reason':str(exc)}


def evaluate_auto(model,rows):
    results=[]
    for r in rows:
        auto=automatic(model,r['prompt'],64)
        base=model.generate(r['prompt']+'\n',max_tokens=64)
        expected={(c.tower,c.round,c.depends_on) for c in oracle(r).calls}
        actual={(c['tower'],c['round'],tuple(c['depends_on'])) for c in auto['calls']}
        results.append({'prompt':r['prompt'],'expected':r['target'],**auto,'coordinator_only':base,
                        'route_correct':actual==expected,'answer_correct':auto['answer'].strip()==r['target'].strip(),
                        'coordinator_correct':base.strip()==r['target'].strip()})
    n=max(1,len(results))
    return {'route_accuracy':sum(r['route_correct'] for r in results)/n,
            'automatic_accuracy':sum(r['answer_correct'] for r in results)/n,
            'coordinator_accuracy':sum(r['coordinator_correct'] for r in results)/n,
            'planner_validity':sum(r['planner_valid'] for r in results)/n,'samples':results,
            'scope':'Small held-out bounded traces, predicted routes and requests; no routing labels supplied to inference'}


def train_delegation(model,cycle,writer):
    from .training import train,make_plan
    rows=traces((cycle%8)*32,32)
    panel=traces(1500,2)
    model.config.active=TOWERS
    before=evaluate_auto(model,panel)
    # Train route choice, round and dependency labels separately from answer labels.
    stages=[]
    from .data import example
    routing_rows=rows+[{**example(t,i,'en'),'source':SOURCE} for t in TOWERS for i in range(8)]
    stages.append(train(model,routing_rows,make_plan('routing',TOWERS,routing_rows,verifier=verified),50,status=writer,verifier=verified))
    plans=[{**r,'prompt':planner_prompt(r['prompt'],oracle(r)), 'target':canonical(r['routing']['requests'])} for r in rows]
    stages.append(train(model,plans,make_plan('planning',TARGETS,plans,verifier=verified),25,status=writer,verifier=verified))
    # Explicit local subtask supervision, then differentiable message/synthesis learning.
    for t in TARGETS:
        local=[r for r in rows if r['tower']==t and not r.get('specialist_targets')]
        replay=[example(t,i,'en') for i in range(32)]
        stages.append(train(model,local,make_plan('continual',(t,),local,verifier=verified),25,replay=replay,status=writer,verifier=verified))
    synthesis=rows+[{**r,'routing':{'targets':[],'rounds':{},'requests':{}}} for r in plans]
    stages.append(train(model,synthesis,make_plan('synthesis',TARGETS,synthesis,verifier=verified),50,status=writer,verifier=verified))
    after=evaluate_auto(model,panel)
    return {'before':before,'after':after,'isolation_checks':all(s['frozen_hashes_verified'] for s in stages),
            'stages':['routing','planning','three specialists','synthesis'],'experimental':True},stages[-1]
