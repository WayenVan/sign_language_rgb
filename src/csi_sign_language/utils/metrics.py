import Levenshtein as L
from typing import *
#calculate levenshtein_distance

def leven_dist(ref, hyp):
    m = len(ref)
    n = len(hyp)
    # 初始化第 1 行
    row = [j for j in range(n + 1)]
    for i in range(1, m + 1):
        # pre 等价于 d[i-1][j-1]
        pre = row[0]
        row[0] = i
        for j in range(1, n + 1):
            # 在给 row[j] 赋值前，row[j] 等价于 d[i-1][j], row[j-1] 等价于 d[i][j-1]
            tmp = row[j]
            if ref[i - 1] == hyp[j - 1]:
                row[j] = pre
            else:
                row[j] = min(row[j], row[j - 1], pre) + 1
            pre = tmp
    return row[n]


def wer(refs: List[List[Hashable]], hyps: List[List[Hashable]]):
    """
    :param refs: list of reference sequences
    :param hyps: list of hyposisthesis
    """
    assert len(refs) == len(hyps)
    n = sum([len(ref) for ref in refs])
    total_levenshtein = sum([L.distance(ref, hyp) for ref, hyp in list(zip(refs, hyps))])

    return total_levenshtein/n
