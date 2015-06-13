
import unittest

import theano
import numpy as np

from theano import tensor as T

import attention

floatX = theano.config.floatX


def test_batched_dot():
    a = T.ftensor3('a')
    b = T.ftensor3('b')

    c = attention.my_batched_dot(a, b)

    # Test in with values
    dim1, dim2, dim3, dim4 = 10, 12, 15, 20

    A_shape = (dim1, dim2, dim3)
    B_shape = (dim1, dim3, dim4)
    C_shape = (dim1, dim2, dim4)

    A = np.arange(np.prod(A_shape)).reshape(A_shape).astype(floatX)
    B = np.arange(np.prod(B_shape)).reshape(B_shape).astype(floatX)

    C = c.eval({a: A, b: B})

    # check shape
    assert C.shape == C_shape

    # check content 
    C_ = np.zeros((dim1, dim2, dim4))
    for i in range(dim1):
        C_[i] = np.dot(A[i], B[i])
    assert np.allclose(C, C_)
    

class TestZoomableAttentionWindow:
    def SetUp(self):
        pass

