from libcpp.vector cimport vector
from libcpp cimport bool
import pickle
import os
import numpy as np
cimport numpy as np


cdef extern from "stdint.h" nogil:
    ctypedef unsigned char  uint8_t


cdef extern from "myXor.hpp":
    void std_vec_xor(uint8_t *, uint8_t *, int)
    void std_vec_minus(float *, float *, float, int);


def myXOR(vec1, vec2, factor, dtype='float64'):
    if dtype == 'uint8':
        return mySTDXOR(vec1, vec2)
    else:
        assert dtype == 'float32'
        return myMinus(vec1, vec2, factor)


def mySTDXOR(vec1, vec2):
    cdef np.ndarray[uint8_t, mode="c"] a = vec1
    cdef np.ndarray[uint8_t, mode="c"] b = vec2
    std_vec_xor(&a[0], &b[0], len(vec1))
    return a
        # return a
def myMinus(vec1, vec2, factor):
    cdef np.ndarray[float, mode="c"] a = vec1
    cdef np.ndarray[float, mode="c"] b = vec2
    std_vec_minus(&a[0], &b[0], factor, len(vec1))
    return a
