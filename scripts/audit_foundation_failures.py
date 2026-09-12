import ast,json,operator,re
from fractions import Fraction
from pathlib import Path
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod}
def calc(text):
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value)==int:return Fraction(node.value)
        if isinstance(node,ast.BinOp) and type(node.op) in OPS:return OPS[type(node.op)](visit(node.left),visit(node.right))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -visit(node.operand)
        raise ValueError('Unsupported expression')
    return visit(ast.parse(text,mode='eval').body)
def cmp(a,b):return '<' if a<b else '>' if a>b else '='
def equations(text):
    work=re.search(r'<work>(.*?)</work>',text).group(1)
    errors=[]
    for step in work.split(';'):
        match=re.fullmatch(r'cmp\((-?\d+),(-?\d+)\)=([<=>])',step)
        try:
            valid=cmp(int(match[1]),int(match[2]))==match[3] if match else calc(step.split('=')[0])==calc(step.split('=')[1])
        except Exception:valid=False
        if not valid:errors.append(step)
    return errors
root=Path('G:/ctfn-text/artifacts/v3_1/math');r=json.loads((root/'y1_add_sub_fluency_promotion_validation.json').read_text());report={'round':r['epoch'],'gold_equation_errors':[],'failures':[]}
for kind in ('active','retention'):
    for s in r[kind]['samples']:
        bad=equations(s['expected_trace'])
        if bad:report['gold_equation_errors'].append({'prompt':s['prompt'],'errors':bad})
        if not s['answer_correct'] or not s['trace_correct']:
            report['failures'].append({'kind':kind,'prompt':s['prompt'],'expected':s['expected'],'output':s['output'],'answer_correct':s['answer_correct'],'invalid_model_equations':equations(s['output'])})
report['equivalent_trace_only_failures']=sum(s['answer_correct'] and not s['invalid_model_equations'] for s in report['failures'])
(root/'foundation_failure_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
