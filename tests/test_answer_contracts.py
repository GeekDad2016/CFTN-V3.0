import pytest
from cftn_v3.answer_contracts import canonical_answer, correct, validated_target
from cftn_v3.learning_experiment import record


def test_math_rejects_truncation_and_prose():
    row=record('math','How much remains?','<answer>65</answer>','test')
    for output,complete in [('<answer>65</answer',False),('The answer is 65',True),('<answer>66</answer>',True)]:
        checked=validated_target(row,output,complete)
        assert checked['target']=='<answer>65</answer>'
        assert not checked['teacher_accepted']
    assert validated_target(row,'<answer>65.0</answer>')['teacher_accepted']
    assert correct('math','<answer>0.5</answer>','1/2')
    assert not correct('math','3 200</answer>','65')


def test_code_equivalence_without_execution():
    reference='def solve(x):\n    return x * 3 + 4'
    assert correct('code','def solve(x):\n return 4 + 3*x',reference)
    assert not correct('code','def solve(x):\n return 4 + 2*x',reference)
    for text in ['import os\ndef solve(x): return x',
                 'def solve(x): return __import__("os").system("echo bad")',
                 '@evil\ndef solve(x): return x',
                 'def solve(x=evil()): return x',
                 'def solve(x):\n while True: pass']:
        with pytest.raises(ValueError):canonical_answer('code',text)


def test_bool_and_proof():
    assert correct('retrieval','YES','yes')
    assert not correct('retrieval','yes, because...','yes')
    assert correct('formal_logic','P(2); Q(2); R(2)','P(2);Q(2);R(2)')
    assert not correct('formal_logic','P(2);Q(3);R(2)','P(2);Q(2);R(2)')
