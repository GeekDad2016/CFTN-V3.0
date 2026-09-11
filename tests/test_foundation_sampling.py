from collections import Counter
from cftn_v3.criterion_sampling import BalancedSampler,case
from cftn_v3.full_curriculum_data import make

def test_small_equality_not_diluted_by_large_numbers():
    rows=[make({'op':'compare','left':n,'right':n},0,'compare') for n in [2,20,200,*range(2000,3000)]]
    sampled=BalancedSampler(rows).sample(300,5)
    assert Counter(case(r).split('/')[0] for r in sampled)=={'small_0_100':100,'medium_101_1000':100,'large_1001_plus':100}
