# import binary_code as bc
import numpy as np


def rlnc(n, k):
    return (n - k) * k / 2


def lt(n, k):
    return n * np.log(k)


n = 100
k = range(5, int(n / 2), 5)
print([rlnc(n, kk) for kk in k])
print([lt(n, kk) for kk in k])
