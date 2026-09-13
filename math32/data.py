import collections, hashlib, json, random
from pathlib import Path
from cftn_v3.math_procedures import solve
from cftn_v3.config import canonical
from cftn_v3.file_io import atomic_json

ROOT=Path('G:/ctfn-text/data/v3_2')
SOURCE=Path('G:/ctfn-text/data/v3_1_subskill_v5')

def identity(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def carries(a,b):
    if a<0 or b<0:raise ValueError('Carry count is defined for nonnegative addition')
    count=chain=longest=carry=0
    while a or b:
        carry=(a%10+b%10+carry)//10;count+=carry;chain=chain+1 if carry else 0;longest=max(longest,chain)
        a//=10;b//=10
    return count,longest

def case(ir):
    op=ir['op'];a=ir.get('left',ir.get('value',0));b=ir.get('right',0)
    if op=='compare_expressions':return ir['left_op']+'/'+ir['right_op']+'/'+solve(ir)[0]
    if isinstance(a,int) and isinstance(b,int):
        band=f'{len(str(abs(a)))}x{len(str(abs(b)))}'
        if op=='add':
            if a<0 or b<0:return band+'/signed/'+('negative_result' if a+b<0 else 'nonnegative_result')
            n,c=carries(a,b);return f'{band}/carry_{n}/chain_{c}'
        if op in ('subtract','borrow_decision','column_subtract'):return band+('/borrow' if a%10<b%10 else '/no_borrow')
        if op=='compare':return band+'/'+('<' if a<b else '>' if a>b else '=')
        if op=='compare_expressions':
            ans=solve(ir)[0];return ir['left_op']+'/'+ir['right_op']+'/'+ans
        if op in ('decompose','regroup'):return 'ones_zero' if a%10==0 else 'ones_nonzero'
        if 'left' in ir and 'right' in ir:return band+('/zero' if not a or not b else '/nonzero')
    numbers=[]
    def collect(v):
        if isinstance(v,(int,float)):numbers.append(v)
        elif isinstance(v,list):
            for x in v:collect(x)
    for v in ir.values():collect(v)
    maximum=max(map(abs,numbers),default=0)
    return ('small' if maximum<10 else 'medium' if maximum<100 else 'large')+('/negative' if any(x<0 for x in numbers) else '/nonnegative')

def answer(ir):
    op=ir['op'];a=ir.get('left',ir.get('value',0));b=ir.get('right',0)
    if op=='decompose':return f'[{a//10},{a%10}]'
    if op=='regroup':return f'[{a//10-1},{a%10+10}]'
    if op=='borrow_decision':return '1' if a%10<b%10 else '0'
    if op=='column_subtract':
        borrow=int(a%10<b%10);units=a%10+10*borrow-b%10;tens=a//10-borrow-b//10
        return f'[{units},{tens},{a-b}]'
    if op=='normalize_fraction':
        from fractions import Fraction
        return str(Fraction(ir['numerator'],ir['denominator']))
    if op=='evaluate_polynomial':return str(sum(c*ir['point']**i for i,c in enumerate(ir['coefficients'])))
    return solve(ir)[0]

def family(ir):
    ir={k:v for k,v in ir.items() if k not in ('type','detail','strategy')}
    if ir['op'] in ('add','multiply') and 'left' in ir:ir={**ir,'left':min(ir['left'],ir['right']),'right':max(ir['left'],ir['right'])}
    if ir['op']=='compare':ir={**ir,'left':min(ir['left'],ir['right']),'right':max(ir['left'],ir['right'])}
    if ir['op']=='compare_expressions':
        from cftn_v3.balanced_curriculum_data import comparison_family
        return comparison_family(ir)
    return identity(ir)

def build():
    assert not ROOT.exists(), 'Immutable dataset already exists'
    ROOT.mkdir(parents=True)
    stages=[];records={s:[] for s in ('train','validation','test')};seen=set();assignments={};audit=[]
    def add(name,scope,requests,origin=None):
        idx=len(stages);name=f'{idx+1:02d}_'+name.split('_',1)[1];buckets=collections.defaultdict(list)
        for ir in requests:
            ir={k:v for k,v in ir.items() if k not in ('type','strategy','detail')}
            key=identity(ir)
            if key in seen:continue
            seen.add(key);f=family(ir)
            row={'ir':ir,'answer':answer(ir),'stage':idx,'criterion':name,'id':key,'family':f,'case':case(ir),'origin':origin or 'v32_generator'}
            buckets[row['case']].append(row)
        assert buckets, name
        counts=collections.Counter();split_cases={s:collections.Counter() for s in records}
        for label,rows in sorted(buckets.items()):
            random.Random(3201+idx).shuffle(rows)
            families=list(dict.fromkeys(r['family'] for r in rows));fresh=[f for f in families if f not in assignments]
            random.Random(3202+idx).shuffle(fresh)
            # Ensure each finite case has held-out coverage when its domain permits it.
            n=len(fresh);nv=max(1,n//10) if n>=5 else 0;nt=nv
            for j,f in enumerate(fresh):assignments[f]='validation' if j<nv else 'test' if j<nv+nt else 'train'
            for row in rows:
                split=assignments[row['family']];records[split].append(row);counts[split]+=1;split_cases[split][label]+=1
        assert counts['train'] and counts['validation'] and counts['test'],(name,counts)
        stages.append({'index':idx,'name':name,'scope':scope,'prerequisite':stages[-1]['name'] if stages else None,
                       'train_records':counts['train'],'validation_records':counts['validation'],'test_records':counts['test']})
        audit.append({'stage':name,'counts':dict(counts),'cases':{s:dict(c) for s,c in split_cases.items()},
                      'sampling':'Uniform stage, then computational case, then unique training record'})
        print(name,dict(counts),flush=True)
    pairs=lambda op,limit,predicate:({'op':op,'left':a,'right':b} for a in range(limit) for b in range(limit) if predicate(a,b))
    add('01_add_within_9','Single-digit addition with result at most 9; direct answer only',pairs('add',10,lambda a,b:a+b<=9))
    add('02_subtract_digits','Single-digit nonnegative subtraction; no regrouping',pairs('subtract',10,lambda a,b:a>=b))
    add('03_tens_and_ones','Decompose a two-digit number into [tens,ones]',({'op':'decompose','value':n} for n in range(10,100)))
    add('04_exchange_one_ten','Exchange one ten: return [remaining_tens,ones_after_exchange]',({'op':'regroup','value':n} for n in range(10,100)))
    add('05_decide_borrow','Decide whether units subtraction needs regrouping: 0=no,1=yes',pairs('borrow_decision',100,lambda a,b:10<=a and 1<=b<=9))
    add('06_add_two_digits_no_carry','Two-digit addition without a carry; direct result',pairs('add',100,lambda a,b: a>=10 and b>=10 and a%10+b%10<10 and a+b<100))
    add('07_add_across_ten','Addition across ten; direct result, no procedural boilerplate',pairs('add',100,lambda a,b:a%10+b%10>=10 and a+b<100))
    add('08_subtract_across_ten','Two-digit minus one-digit subtraction; direct result',pairs('subtract',21,lambda a,b:a>=10 and 1<=b<=9))
    add('09_column_subtraction','Explicit [units_result,tens_result,whole_result], after regrouping prerequisites',pairs('column_subtract',100,lambda a,b:10<=a and b<=a))
    add('10_subtract_within_99','Two-digit subtraction; direct result',pairs('subtract',100,lambda a,b:a>=10 and b>=10 and a>=b))
    add('11_multiply_digits','Single-digit multiplication',pairs('multiply',10,lambda a,b:True))
    add('12_multiply_two_digits','Two-digit by one-digit multiplication',pairs('multiply',100,lambda a,b:10<=a and b<=9))
    add('13_exact_division','Exact nonnegative division with divisor 1 to 9',({'op':'divide','dividend':a*b,'divisor':b} for a in range(100) for b in range(1,10)))
    add('14_number_comparison','Compare numbers 0 to 99',pairs('compare',100,lambda a,b:True))
    rng=random.Random(3232)
    def expressions():
        for _ in range(20000):
            ir={'op':'compare_expressions'}
            for s in ('left','right'):
                a,b=rng.randrange(21),rng.randrange(21);op=rng.choice(('add','subtract'))
                if op=='subtract':a,b=max(a,b),min(a,b)
                ir[s]=[a,b];ir[s+'_op']=op
            yield ir
    add('15_compare_expression_results','Compute two small expressions then compare their results',expressions())
    add('00_successor','Count forward by one within 100',({'op':'successor','value':a} for a in range(100)))
    add('00_predecessor','Count backward by one within 100',({'op':'predecessor','value':a} for a in range(1,101)))
    add('00_missing_count_sequence','Complete consecutive counting sequences within 100',({'op':'missing_count_sequence','sequence':[a,a+1,None,a+3],'missing_index':2} for a in range(97)))
    add('00_compose_place_value','Compose a number from tens and ones',({'op':'compose_place_value','tens':a,'ones':b} for a in range(10) for b in range(10)))
    add('00_neighbouring_tens','Find the neighbouring multiples of ten',({'op':'neighbouring_tens','value':a} for a in range(1,100)))
    add('00_compare_place_value','Compare a tens-and-ones representation with an integer',({'op':'compare_place_value','tens':a//10,'ones':a%10,'right':b} for a in range(100) for b in range(100)))
    add('00_missing_addend','Find a missing addend in sums within 99',({'op':'missing_addend','known':a,'total':a+b} for a in range(100) for b in range(100-a)))
    add('00_missing_minuend','Find a missing starting value in subtraction within 99',({'op':'missing_minuend','right':b,'result':a-b} for a in range(100) for b in range(a+1)))
    add('00_missing_subtrahend','Find the amount subtracted within 99',({'op':'missing_subtrahend','left':a,'result':a-b} for a in range(100) for b in range(a+1)))
    add('00_difference','Absolute difference of two values within 99',pairs('difference',100,lambda a,b:True))
    # Preserve later curriculum breadth, separating each operation into its own stage.
    source=[json.loads(l) for l in (SOURCE/'train.jsonl').read_text().splitlines()]
    old=json.loads((SOURCE/'manifest.json').read_text())
    add('16_signed_addition','Addition of signed single-digit integers',({'op':'add','left':a,'right':b} for a in range(-9,10) for b in range(-9,10)))
    add('17_fraction_representation','Reduce a positive fraction to canonical numerator/denominator',({'op':'normalize_fraction','numerator':a,'denominator':b} for a in range(21) for b in range(1,21)))
    add('18_polynomial_values','Evaluate a linear or quadratic polynomial from coefficients',({'op':'evaluate_polynomial','coefficients':[a,b,c],'point':x} for a in range(-3,4) for b in range(-3,4) for c in range(3) for x in range(-3,4)))
    order='add subtract multiply divide fraction_add percent_of rectangle_area power linear_solve pythagoras arithmetic_sequence quadratic_roots quadratic_general simultaneous_solve differentiate_monomial differentiate_polynomial polynomial_derivative_value integrate_monomial definite_integral_polynomial binomial_coefficient combination permutation binomial_probability_half binomial_probability uniform_expectation weighted_expectation constant_acceleration cancelled_linear_limit matrix_det_2x2 matrix_det_3x3 triangular_eigenvalues eigenvalues_2x2 euclid_gcd_invariant bezout_coefficients mod_inverse cyclic_element_order permutation_order geometric_series_radius geometric_series_point square_identity polynomial_identity odd_square_counterexample'.split()
    groups=collections.defaultdict(list)
    for r in source:
        if r['stage']>=5:groups[r['ir']['op']].append(r['ir'])
    assert set(groups)<=set(order),set(groups)-set(order)
    for op in order:
        requests=groups.get(op,[])
        if requests:
            unique={identity({k:v for k,v in ir.items() if k not in ('type','strategy','detail')}):ir for ir in requests}
            novel=[ir for k,ir in unique.items() if k not in seen]
            if len(novel)<30:continue
            add(f'{len(stages)+1:02d}_{op}',f'{op}: one operation at a time, bounded synthetic inputs',novel,'v31_train_only_repartitioned')
    for split,rows in records.items():
        (ROOT/(split+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
    for x in records:
        for y in records:
            if x!=y:assert not {r['family'] for r in records[x]}&{r['family'] for r in records[y]}
    manifest={'version':'3.2','stages':stages,'records':{s:len(r) for s,r in records.items()},
              'files':{s+'.jsonl':hashlib.sha256((ROOT/(s+'.jsonl')).read_bytes()).hexdigest() for s in records}}
    atomic_json(ROOT/'manifest.json',manifest);atomic_json(ROOT/'audit.json',audit)
    print(json.dumps({'stages':len(stages),'records':manifest['records']}))

if __name__=='__main__':build()
