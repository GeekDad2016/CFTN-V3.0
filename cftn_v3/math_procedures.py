"""Exact, bounded Maths procedures and a single operation-aware answer contract.

No generated code is executed. Targets are reconstructed from typed inputs.
"""
import json
import math
import re
from fractions import Fraction as Q
from .config import canonical


def fmt(x):
    if isinstance(x,bool):return str(x).lower()
    if isinstance(x,Q):return str(x)
    if isinstance(x,(tuple,list)):return '['+','.join(fmt(v) for v in x)+']'
    return str(x)


def determinant(a):
    if len(a)==1:return Q(a[0][0])
    return sum(((-1)**j*Q(a[0][j])*determinant([r[:j]+r[j+1:] for r in a[1:]]) for j in range(len(a))),Q(0))


def polynomial(a,x):return sum((Q(c)*Q(x)**i for i,c in enumerate(a)),Q(0))


def solve(ir):
    op=ir['op'];steps=[]
    def eq(lhs,result):steps.append(f'{lhs}={fmt(result)}');return result
    def n(x):return f'({x})' if Q(x)<0 else str(x)
    def mul(a,b):
        parts=[]
        for i,d in enumerate(reversed(str(abs(int(b))))):
            if int(d):
                v=int(d)*10**i*(-1 if b<0 else 1);parts.append(eq(f'{n(a)}*{n(v)}',a*v))
        if not parts:return eq(f'{n(a)}*0',0)
        return eq('+'.join(n(v) for v in parts),sum(parts)) if len(parts)>1 else parts[0]
    if op in ('successor','predecessor'):
        d=1 if op=='successor' else -1;answer=eq(f"{ir['value']}+({d})",ir['value']+d)
    elif op=='missing_count_sequence':
        a=ir['sequence'][ir['missing_index']-1];answer=eq(f'{a}+1',a+1)
    elif op=='compare':
        a,b=ir['left'],ir['right'];answer=eq(f'cmp({a},{b})','<' if a<b else '>' if a>b else '=')
    elif op=='add':
        values=ir.get('operands',[ir.get('left'),ir.get('right')]);answer=values[0]
        for v in values[1:]:answer=eq(f'{n(answer)}+{n(v)}',answer+v)
    elif op in ('subtract','difference'):
        a,b=ir['left'],ir['right'];answer=eq(f'{n(a)}-{n(b)}',a-b)
        if op=='difference':answer=eq(f'abs({answer})',abs(answer))
    elif op=='missing_addend':answer=eq(f"{ir['total']}-{ir['known']}",ir['total']-ir['known'])
    elif op=='place_value':answer=eq(f"divmod({ir['value']},10)",list(divmod(ir['value'],10)))
    elif op=='neighbouring_tens':
        a=eq(f"10*floor({ir['value']}/10)",ir['value']//10*10);b=eq(f'{a}+10',a+10);answer=[a,b]
    elif op=='add_signed':answer=eq(f"{n(ir['value'])}+{n(ir['delta'])}",ir['value']+ir['delta'])
    elif op=='multiply':answer=mul(ir['left'],ir['right'])
    elif op=='divide':answer=eq(f"{ir['dividend']}/{n(ir['divisor'])}",Q(ir['dividend'],ir['divisor']))
    elif op=='fraction_add':
        a,b=ir['left'];c,d=ir['right'];num=eq(f'{a}*{d}+{c}*{b}',a*d+c*b);den=eq(f'{b}*{d}',b*d);answer=eq(f'{num}/{den}',Q(num,den))
    elif op=='percent_of':
        v=mul(ir['percent'],ir['base']);answer=eq(f'{v}/100',Q(v,100))
    elif op=='rectangle_area':answer=mul(ir['width'],ir['height'])
    elif op=='linear_solve':
        v=eq(f"{ir['c']}-{n(ir['b'])}",ir['c']-ir['b']);answer=eq(f"{v}/{n(ir['a'])}",Q(v,ir['a']))
    elif op=='power':
        b,e=ir['base'],ir['exponent'];answer=b
        if e==0:answer=eq(f'{n(b)}^0',1)
        for _ in range(1,e):answer=eq(f'{n(answer)}*{n(b)}',answer*b)
    elif op=='pythagoras':
        a,b=ir['a'],ir['b'];aa=eq(f'{a}^2',a*a);bb=eq(f'{b}^2',b*b);s=eq(f'{aa}+{bb}',aa+bb)
        answer=math.isqrt(s)
        if answer*answer!=s:raise ValueError('non-integral hypotenuse outside contract')
        eq(f'sqrt({s})',answer)
    elif op in ('quadratic_roots','quadratic_general'):
        a=ir.get('a',1);b,c=ir['b'],ir['c'];d=eq(f'{n(b)}^2-4*{n(a)}*{n(c)}',b*b-4*a*c);s=math.isqrt(d)
        if s*s!=d:raise ValueError('non-rational roots outside contract')
        eq(f'sqrt({d})',s);answer=sorted([eq(f'({-b}-{s})/(2*{n(a)})',Q(-b-s,2*a)),eq(f'({-b}+{s})/(2*{n(a)})',Q(-b+s,2*a))])
    elif op=='simultaneous_solve':
        (a,b,e),(c,d,f)=ir['equations'];det=eq(f'{n(a)}*{n(d)}-{n(b)}*{n(c)}',a*d-b*c)
        x=eq(f'({n(e)}*{n(d)}-{n(b)}*{n(f)})/{n(det)}',Q(e*d-b*f,det))
        y=eq(f'({n(a)}*{n(f)}-{n(e)}*{n(c)})/{n(det)}',Q(a*f-e*c,det));answer=[x,y]
    elif op=='arithmetic_sequence':
        v=eq(f"({ir['index']}-1)*{n(ir['difference'])}",(ir['index']-1)*ir['difference']);answer=eq(f"{n(ir['first'])}+{n(v)}",ir['first']+v)
    elif op in ('differentiate_monomial','integrate_monomial'):
        c,p=ir['coefficient'],ir['power']
        if op=='differentiate_monomial':v=eq(f'{n(c)}*{p}',c*p);answer=f'{v}*x^{p-1}';eq(f'D({c}*x^{p})',answer)
        else:v=eq(f'{n(c)}/({p}+1)',Q(c,p+1));answer=f'{fmt(v)}*x^{p+1}+C';eq(f'integral({c}*x^{p})',answer)
    elif op in ('combination','binomial_coefficient','permutation'):
        n_,k=ir['n'],ir['k'];a=eq(f'{n_}!/{n_-k}!',math.perm(n_,k))
        answer=a if op=='permutation' else eq(f'{a}/{k}!',a//math.factorial(k))
    elif op in ('binomial_probability_half','binomial_probability'):
        n_,k=ir['trials'],ir['successes'];p=Q(*ir.get('probability',[1,2]));v=eq(f'C({n_},{k})',math.comb(n_,k))
        answer=eq(f'{v}*({fmt(p)})^{k}*({fmt(1-p)})^{n_-k}',v*p**k*(1-p)**(n_-k))
    elif op=='constant_acceleration':
        u,a,t=ir['u'],ir['a'],ir['t'];v=eq(f'{n(u)}*{t}',u*t);w=eq(f'{n(a)}*{t}^2/2',Q(a*t*t,2));answer=eq(f'{fmt(v)}+({fmt(w)})',v+w)
    elif op in ('matrix_det_2x2','matrix_det_3x3'):
        a=ir['matrix']
        if len(a)==2:answer=eq(f'{n(a[0][0])}*{n(a[1][1])}-{n(a[0][1])}*{n(a[1][0])}',determinant(a))
        else:
            terms=[]
            for j in range(3):
                minor=eq(f'minor(0,{j})',determinant([r[:j]+r[j+1:] for r in a[1:]]));terms.append(eq(f'(-1)^{j}*{n(a[0][j])}*{n(minor)}',(-1)**j*a[0][j]*minor))
            answer=eq('+'.join(f'({fmt(v)})' for v in terms),sum(terms))
    elif op=='cancelled_linear_limit':answer=eq('lim((m*x+b-(m*a+b))/(x-a))',ir['slope'])
    elif op in ('mod_inverse','bezout_coefficients'):
        a,m=ir['value'],ir['modulus'];r0,r1=a,m;s0,s1=1,0;t0,t1=0,1
        while r1:
            q=r0//r1;eq(f'{r0} mod {r1}',r0-q*r1)
            r0,r1=r1,r0-q*r1;s0,s1=s1,s0-q*s1;t0,t1=t1,t0-q*t1
        eq(f'{a}*{s0}+{m}*{t0}',r0)
        if op=='mod_inverse':
            if r0!=1:raise ValueError('inverse does not exist')
            answer=eq(f'{s0} mod {m}',s0%m)
        else:answer=[r0,s0,t0]
    elif op=='uniform_expectation':answer=eq(f"({ir['minimum']}+{ir['maximum']})/2",Q(ir['minimum']+ir['maximum'],2))
    elif op=='weighted_expectation':
        terms=[eq(f'{v}*({w[0]}/{w[1]})',v*Q(*w)) for v,w in zip(ir['values'],ir['weights'])]
        if sum(Q(*w) for w in ir['weights'])!=1:raise ValueError('probabilities do not sum to one')
        answer=eq('+'.join(f'({fmt(v)})' for v in terms),sum(terms))
    elif op=='geometric_series_radius':answer=eq(f"1/{ir['coefficient_base']}",Q(1,ir['coefficient_base']))
    elif op=='geometric_series_point':
        ratio=eq(f"({fmt(Q(*ir['ratio']))})*({fmt(Q(*ir['point']))})",Q(*ir['ratio'])*Q(*ir['point']))
        answer=eq(f'abs({fmt(ratio)})<1',abs(ratio)<1)
    elif op=='cyclic_element_order':
        g=eq(f"gcd({ir['element']},{ir['modulus']})",math.gcd(ir['element'],ir['modulus']));answer=eq(f"{ir['modulus']}/{g}",ir['modulus']//g)
    elif op=='permutation_order':
        p=ir['permutation'];visited=set();sizes=[]
        if sorted(p)!=list(range(len(p))):raise ValueError('invalid permutation')
        for i in range(len(p)):
            if i in visited:continue
            cycle=[];j=i
            while j not in visited:visited.add(j);cycle.append(j);j=p[j]
            sizes.append(len(cycle));eq(f'cycle{fmt(cycle)}',len(cycle))
        answer=eq('lcm'+fmt(sizes),math.lcm(*sizes))
    elif op in ('triangular_eigenvalues','eigenvalues_2x2'):
        a=ir['matrix'];tr=eq('trace(A)',a[0][0]+a[1][1]);det=eq('det(A)',determinant(a));d=eq(f'{tr}^2-4*({fmt(det)})',tr*tr-4*det)
        s=math.isqrt(int(d))
        if Q(s*s)!=d:raise ValueError('non-rational spectrum outside contract')
        answer=sorted([eq(f'({tr}-{s})/2',Q(tr-s,2)),eq(f'({tr}+{s})/2',Q(tr+s,2))])
    elif op=='square_identity':
        a,b=ir['a'],ir['b'];left=eq(f'({n(a)}+{n(b)})^2',(a+b)**2);right=eq(f'{n(a)}^2+2*{n(a)}*{n(b)}+{n(b)}^2',a*a+2*a*b+b*b);answer=eq(f'{left}=={right}',left==right)
    elif op=='odd_square_counterexample':
        v=ir['value'];sq=eq(f'{v}^2',v*v);eq(f'{sq} mod 2',sq%2);answer=[v,sq]
    elif op=='euclid_gcd_invariant':
        a,b=ir['a'],ir['b'];r=eq(f'{a} mod {b}',a%b);g=eq(f'gcd({a},{b})',math.gcd(a,b));h=eq(f'gcd({b},{r})',math.gcd(b,r));answer=eq(f'{g}=={h}',g==h)
    elif op=='differentiate_polynomial':
        answer=[eq(f'{i}*({c})',i*c) for i,c in enumerate(ir['coefficients']) if i]
    elif op=='definite_integral_polynomial':
        a,b=ir['lower'],ir['upper'];terms=[]
        for i,c in enumerate(ir['coefficients']):terms.append(eq(f'({c})*({n(b)}^{i+1}-{n(a)}^{i+1})/{i+1}',Q(c*(b**(i+1)-a**(i+1)),i+1)))
        answer=eq('+'.join(f'({fmt(v)})' for v in terms),sum(terms))
    elif op=='polynomial_derivative_value':
        coeff=[i*c for i,c in enumerate(ir['coefficients']) if i];eq('D_coefficients',coeff);answer=eq(f"D_at({ir['point']})",polynomial(coeff,ir['point']))
    elif op=='polynomial_identity':
        a,b=ir['left'],ir['right'];size=max(len(a),len(b));a=a+[0]*(size-len(a));b=b+[0]*(size-len(b))
        residual=eq('left_coefficients-right_coefficients',[x-y for x,y in zip(a,b)]);answer=eq('all_residuals_zero',all(v==0 for v in residual))
    else:raise ValueError('Unsupported mathematical procedure: '+op)
    return fmt(answer),'<work>'+';'.join(steps)+'</work><answer>'+fmt(answer)+'</answer>'


TUPLE_OPS={'place_value','neighbouring_tens','quadratic_roots','quadratic_general','simultaneous_solve',
 'triangular_eigenvalues','eigenvalues_2x2','odd_square_counterexample','bezout_coefficients','differentiate_polynomial'}


def normalize_answer(text,op):
    text=''.join(text.split())
    if op in TUPLE_OPS:
        body=text[1:-1] if text.startswith('[') and text.endswith(']') else text
        return '['+','.join(str(Q(x)) for x in body.split(','))+']'
    try:return str(Q(text))
    except ValueError:return text


def score(output,row,complete=True):
    matches=re.findall(r'<answer>(.*?)</answer>',output,re.S)
    valid=len(matches)==1 and output.rstrip().endswith('</answer>')
    try:correct=valid and normalize_answer(matches[0],row['ir']['op'])==row['answer']
    except (ValueError,ZeroDivisionError):correct=False
    # Comparisons legitimately contain < or > inside the worked expression.
    structured=bool(re.fullmatch(r'<work>(?:(?!</?work>|</?answer>).)*</work><answer>.*?</answer>\s*',output,re.S))
    return {'answer_correct':bool(correct),'format_correct':bool(valid and complete and structured),
        'trace_correct':bool(complete and ''.join(output.split())==''.join(row['target'].split()))}
