"""Bounded Stage 2 generation verifier and supervised failure replay."""
import ast
import operator
import re
from .criterion_sampling import BalancedSampler

OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul}

def integer_expression(text):
    if len(text)>200:raise ValueError('Expression too long')
    tree=ast.parse(text,mode='eval')
    if sum(1 for _ in ast.walk(tree))>80:raise ValueError('Expression too complex')
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value)==int and abs(n.value)<10**12:return n.value
        if isinstance(n,ast.BinOp) and type(n.op) in OPS:return OPS[type(n.op)](visit(n.left),visit(n.right))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -visit(n.operand)
        raise ValueError('Unsupported expression')
    return visit(tree.body)

def relation(a,b):return '<' if a<b else '>' if a>b else '='

def expected_answer(ir):
    op=ir['op']
    if op in ('add','subtract'):
        return str(ir['left']+ir['right'] if op=='add' else ir['left']-ir['right'])
    if op=='compare':return relation(ir['left'],ir['right'])
    if op=='compare_expressions':
        values=[sum(ir[s]) if ir[s+'_op']=='add' else ir[s][0]-ir[s][1] for s in ('left','right')]
        return relation(*values)
    if op=='missing_addend':return str(ir['total']-ir['known'])
    if op=='missing_minuend':return str(ir['result']+ir['right'])
    if op=='missing_subtrahend':return str(ir['left']-ir['result'])
    raise ValueError('Unsupported operation')

def verify(row, output):
    """Unknown alternative procedures are not treated as proven mathematical errors."""
    ir=row['ir']; expected=expected_answer(ir)
    assert expected==row['answer'],'Generator disagrees with independent answer checker'
    match=re.fullmatch(r'<work>(.*?)</work><answer>(.*?)</answer>',output.strip(),re.S)
    if not match:return {'needs_correction':True,'kind':'format','feedback':'Use one work section and one answer section.'}
    work,answer=match.groups();steps=work.split(';')
    for i,step in enumerate(steps):
        comparison=re.fullmatch(r'cmp\((-?\d+),(-?\d+)\)=([<=>])',step)
        try:
            valid=(relation(int(comparison[1]),int(comparison[2]))==comparison[3] if comparison
                   else len(step.split('='))==2 and integer_expression(step.split('=')[0])==integer_expression(step.split('=')[1]))
        except (ValueError,SyntaxError,RecursionError):valid=None
        if valid is False:
            return {'needs_correction':True,'kind':'invalid_step','step_index':i,'step':step,
                    'feedback':'This equation or comparison is false. Recompute it; use the verified target as supervision.'}
    if answer.strip()!=expected:
        return {'needs_correction':True,'kind':'wrong_answer','feedback':f'Expected {expected}, received {answer}.'}
    if re.sub(r'\s','',output)==re.sub(r'\s','',row['target']):
        return {'needs_correction':False,'kind':'verified_canonical'}
    # A direct arithmetic equality tied to the original operands is also a valid proof.
    if ir['op'] in ('add','subtract') and len(steps)==1:
        symbol='+' if ir['op']=='add' else '-'
        direct=f"{ir['left']}{symbol}{ir['right']}={expected}"
        if re.sub(r'\s','',work)==direct:return {'needs_correction':False,'kind':'verified_alternative'}
    return {'needs_correction':False,'kind':'unverified_alternative','feedback':'Correct answer, but this verifier cannot establish trace relevance; excluded from correction mining.'}

def correction_rows(failed, active, prior, count, seed):
    if not failed:raise ValueError('No verified generated failures to correct')
    n=count*3//5; replay=count//5 if prior else 0
    return (BalancedSampler(failed).sample(n,seed)
            +BalancedSampler(active).sample(count-n-replay,seed+1)
            +BalancedSampler(prior).sample(replay,seed+2))
