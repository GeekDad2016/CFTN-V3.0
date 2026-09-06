"""Bounded answer contracts. Never execute generated Python."""
import ast
import re
from fractions import Fraction

CONTRACT_VERSION = 'canonical_v2'
FORMATS = {
    'math': 'Return only <answer>NUMBER</answer>, with no explanation.',
    'retrieval': 'Return exactly yes or no, in lowercase.',
    'code': 'Return only the Python function solve(x). No markdown or explanation.',
    'formal_logic': 'Return only the requested propositions separated by semicolons. No explanation.',
}


def number(text):
    text = text.strip().replace(',', '')
    if not re.fullmatch(r'[+-]?\d+(?:\.\d+)?(?:/\d+)?', text):
        raise ValueError('not a single number')
    return Fraction(text)


def canonical_answer(tower, text):
    text = text.strip()
    if tower == 'math':
        match = re.fullmatch(r'<answer>(.*?)</answer>', text, re.S)
        value = number(match[1] if match else text)
        return f'<answer>{value}</answer>'
    if tower == 'retrieval':
        if text.lower() not in ('yes', 'no'): raise ValueError('not yes/no')
        return text.lower()
    if tower == 'formal_logic':
        text = re.sub(r'\s+', '', text)
        if not re.fullmatch(r'P\(\d+\);Q\(\d+\);R\(\d+\)', text):
            raise ValueError('invalid bounded proof')
        if len(set(re.findall(r'\d+', text))) != 1: raise ValueError('inconsistent proof')
        return text
    if tower == 'code':
        if text.startswith('```'):
            match = re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```', text, re.S)
            if not match: raise ValueError('incomplete code fence')
            text = match[1]
        tree = ast.parse(text)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError('expected one function')
        fn = tree.body[0]
        if fn.name != 'solve' or fn.decorator_list or fn.returns or fn.type_comment:
            raise ValueError('unexpected function definition')
        if (len(fn.args.args) != 1 or fn.args.args[0].arg != 'x' or fn.args.args[0].annotation
                or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg
                or fn.args.defaults or fn.args.kw_defaults):
            raise ValueError('expected solve(x)')
        if len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
            raise ValueError('expected a single arithmetic return')
        def check(node):
            if isinstance(node, ast.Name) and node.id == 'x': return
            if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) < 10**9: return
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                check(node.left); check(node.right); return
            raise ValueError('unsupported expression')
        check(fn.body[0].value)
        return ast.unparse(tree)
    raise ValueError('no contract for tower')


def correct(tower, output, reference):
    try:
        a, b = canonical_answer(tower, output), canonical_answer(tower, reference)
        if tower != 'code': return a == b
        # Compare exact polynomials, not a few execution probes or arbitrary code.
        def polynomial(n):
            if isinstance(n, ast.Name): return {1: 1}
            if isinstance(n, ast.Constant): return {0: n.value} if n.value else {}
            left, right = polynomial(n.left), polynomial(n.right)
            result = dict(left) if not isinstance(n.op, ast.Mult) else {}
            if isinstance(n.op, ast.Mult):
                for i, x in left.items():
                    for j, y in right.items(): result[i+j] = result.get(i+j, 0) + x*y
            else:
                for i, x in right.items(): result[i] = result.get(i, 0) + (-x if isinstance(n.op, ast.Sub) else x)
            return {i:x for i,x in result.items() if x}
        return polynomial(ast.parse(a).body[0].body[0].value) == polynomial(ast.parse(b).body[0].body[0].value)
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
        return False


def validated_target(row, text, complete=True):
    reference = canonical_answer(row['tower'], row['reference'])
    accepted = complete and correct(row['tower'], text, reference)
    return {**row, 'target': canonical_answer(row['tower'], text) if accepted else reference,
            'reference': reference, 'contract_version': CONTRACT_VERSION,
            'teacher_raw': text, 'teacher_accepted': bool(accepted),
            'supervision': 'validated_teacher' if accepted else 'dataset_reference_fallback',
            'rejection_reason': None if accepted else ('truncated' if not complete else 'format_or_reference_mismatch')}
