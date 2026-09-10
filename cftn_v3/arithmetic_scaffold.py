"""Compact place-value arithmetic traces for explicit scaffold requests."""
def solve_scaffold(ir):
    op=ir['op'];steps=[]
    def eq(lhs,value):steps.append(f'{lhs}={value}');return value
    def addition(a,b):
        carry=0;digits=[]
        for i in range(max(len(str(a)),len(str(b)))):
            total=eq(f'{a//10**i%10}+{b//10**i%10}+{carry}',a//10**i%10+b//10**i%10+carry)
            digits.append(total%10);carry=eq(f'{total}//10',total//10)
            eq(f'{total}%10',total%10)
        if carry:digits.append(carry)
        return eq('+'.join(f'{v}*{10**i}' for i,v in enumerate(digits)),a+b)
    if op=='add':answer=addition(ir['left'],ir['right'])
    elif op=='subtract':
        a,b=ir['left'],ir['right'];borrow=0;digits=[]
        for i in range(len(str(a))):
            x=a//10**i%10;y=b//10**i%10;v=x-borrow-y;extra=10 if v<0 else 0
            digits.append(eq(f'{x}-{borrow}+{extra}-{y}',v+extra));borrow=int(v<0)
        answer=eq('+'.join(f'{v}*{10**i}' for i,v in enumerate(digits)),a-b)
    elif op=='divide':
        a,b=ir['dividend'],ir['divisor'];rem=0;quotient=[]
        for digit in str(a):
            value=eq(f'{rem}*10+{digit}',rem*10+int(digit));q=eq(f'{value}//{b}',value//b)
            rem=eq(f'{value}-{q}*{b}',value-q*b);quotient.append(str(q))
        answer=eq(f'{a}/{b}',int(''.join(quotient)))
        eq(f'{answer}*{b}',a)
    elif op=='multiply':
        a,b=ir['left'],ir['right'];parts=[eq(f'{a}*{b%10}',a*(b%10)),eq(f'{a}*{b//10*10}',a*(b//10*10))]
        answer=addition(*parts)
    else:raise ValueError('Unsupported scaffold')
    return str(answer),'<work>'+';'.join(steps)+'</work><answer>'+str(answer)+'</answer>'
