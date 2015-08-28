
import unittest

import theano
import numpy as np

from theano import tensor as T

from draw.attention import *

floatX = theano.config.floatX


def test_batched_dot():
    a = T.ftensor3('a')
    b = T.ftensor3('b')

    c = my_batched_dot(a, b)

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
    def setUp(self):
        # Device under test
        self.channels = 1
        self.height = 50
        self.width = 120
        self.N = 100

        self.zaw = ZoomableAttentionWindow(self.channels, self.height, self.width, self.N)

    def test_filterbank_matrices(self):
        batch_size = 100
        height, width = self.height, self.width
        N = self.N
        zaw = self.zaw

        # Create theano function
        center_y, center_x = T.fvectors('center_x', 'center_y')
        delta, sigma = T.fvectors('delta', 'sigma')

        FY, FX = zaw.filterbank_matrices(center_y, center_x, delta, sigma)

        do_filterbank = theano.function(
            [center_y, center_x, delta, sigma],
            [FY, FX],
            name="do_filterbank_matrices",
            allow_input_downcast=True)

        # test theano function
        center_y = np.linspace(-height, 2*height, batch_size)
        center_x = np.linspace(-width, 2*width, batch_size)
        delta = np.linspace(0.1, height, batch_size)
        sigma = np.linspace(0.1, height, batch_size)

        FY, FX = do_filterbank(center_y, center_x, delta, sigma)

        assert FY.shape == (batch_size, N, height)
        assert FX.shape == (batch_size, N, width)

        assert np.isfinite(FY).all()
        assert np.isfinite(FX).all()

    def test_read(self):
        batch_size = 100
        height, width = self.height, self.width
        N = self.N
        zaw = self.zaw

        # Create theano function
        images = T.ftensor3('images')
        center_y, center_x = T.fvectors('center_x', 'center_y')
        delta, sigma = T.fvectors('delta', 'sigma')

        readout = zaw.read(images, center_y, center_x, delta, sigma)

        do_read = theano.function(
            [images, center_y, center_x, delta, sigma],
            readout,
            name="do_read",
            allow_input_downcast=True)

        # Test theano function
        images = np.random.uniform(size=(batch_size, height, width))
        center_y = np.linspace(-height, 2*height, batch_size)
        center_x = np.linspace(-width, 2*width, batch_size)
        delta = np.linspace(0.1, height, batch_size)
        sigma = np.linspace(0.1, height, batch_size)

        readout = do_read(images, center_y, center_x, delta, sigma)

        assert readout.shape == (batch_size, N**2)
        assert np.isfinite(readout).all()
        assert (readout >= 0.).all()
        assert (readout <= 1.).all()

    def test_write(self):
        batch_size = 100
        height, width = self.height, self.width
        N = self.N
        zaw = self.zaw

        # Create theano function
        content = T.fmatrix('content')
        center_y, center_x = T.fvectors('center_x', 'center_y')
        delta, sigma = T.fvectors('delta', 'sigma')

        images = zaw.write(content, center_y, center_x, delta, sigma)

        do_write = theano.function(
            [content, center_y, center_x, delta, sigma],
            images,
            name="do_write",
            allow_input_downcast=True)

        # Test theano function
        content = np.random.uniform(size=(batch_size, N**2))
        center_y = np.linspace(-height, 2*height, batch_size)
        center_x = np.linspace(-width, 2*width, batch_size)
        delta = np.linspace(0.1, height, batch_size)
        sigma = np.linspace(0.1, height, batch_size)

        images = do_write(content, center_y, center_x, delta, sigma)

        assert images.shape == (batch_size, height*width)
        assert np.isfinite(images).all()
