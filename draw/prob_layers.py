
from __future__ import division, print_function 

import logging

import numpy
import theano

from theano import tensor

from blocks.bricks.base import application, lazy
from blocks.roles import add_role, WEIGHT, BIAS
from blocks.bricks import Random, Initializable, Linear
from blocks.utils import shared_floatx_zeros

logger = logging.getLogger(__name__)
floatX = theano.config.floatX

N_STREAMS = 2048

#-----------------------------------------------------------------------------
 
def logsumexp(A, axis=None):
    """Numerically stable log( sum( exp(A) ) ) """
    A_max = tensor.max(A, axis=axis, keepdims=True)
    B = tensor.log(tensor.sum(tensor.exp(A-A_max), axis=axis, keepdims=True))+A_max
    B = tensor.sum(B, axis=axis)
    return B


def replicate_batch(A, repeat):
    """Extend the given 2d Tensor by repeating reach line *repeat* times.

    With A.shape == (rows, cols), this function will return an array with
    shape (rows*repeat, cols).

    Parameters
    ----------
    A : T.tensor
        Each row of this 2d-Tensor will be replicated *repeat* times
    repeat : int

    Returns
    -------
    B : T.tensor
    """
    A_ = A.dimshuffle((0, 'x', 1))
    A_ = A_ + tensor.zeros((A.shape[0], repeat, A.shape[1]), dtype=floatX)
    A_ = A_.reshape( [A_.shape[0]*repeat, A.shape[1]] )
    return A_

#-----------------------------------------------------------------------------


class ProbabilisticTopLayer(Random):
    def __init__(self, **kwargs):
        super(ProbabilisticTopLayer, self).__init__(**kwargs)

    def sample_expected(self):
        raise NotImplemented

    def sample(self):
        raise NotImplemented

    def log_prob(self, X):
        raise NotImplemented


class ProbabilisticLayer(Random):
    def __init__(self, **kwargs):
        super(ProbabilisticLayer, self).__init__(**kwargs)

    def sample_expected(self, Y):
        raise NotImplemented

    def sample(self, Y):
        raise NotImplemented

    def log_prob(self, X, Y):
        raise NotImplemented

#-----------------------------------------------------------------------------


class BernoulliTopLayer(Initializable, ProbabilisticTopLayer):
    @lazy
    def __init__(self, dim_X, biases_init, **kwargs):
        super(BernoulliTopLayer, self).__init__(**kwargs)
        self.dim_X = dim_X
        self.biases_init = biases_init

    def _allocate(self):
        b = shared_floatx_zeros((self.dim_X,), name='b')
        add_role(b, BIASES)
        self.parameters.append(b)
        self.add_auxiliary_variable(b.norm(2), name='b_norm')
        
    def _initialize(self):
        b, = self.parameters
        self.biases_init.initialize(b, self.rng)


    @application(inputs=[], outputs=['X_expected'])
    def sample_expected(self):
        b = self.parameters[0]
        return tensor.nnet.sigmoid(b)

    @application(outputs=['X', 'log_prob'])
    def sample(self, n_samples):
        prob_X = self.sample_expected()
        U = self.theano_rng.uniform(size=(n_samples, prob_X.shape[0]), nstreams=N_STREAMS)
        X = tensor.cast(U <= prob_X, floatX)
        return X, self.log_prob(X)

    @application(inputs='X', outputs='log_prob')
    def log_prob(self, X):
        prob_X = self.sample_expected()
        log_prob = X*tensor.log(prob_X) + (1.-X)*tensor.log(1.-prob_X)
        return log_prob.sum(axis=1)


class BernoulliLayer(Initializable, ProbabilisticLayer):
    @lazy
    def __init__(self, dim_X, dim_Y, **kwargs):
        super(BernoulliLayer, self).__init__(**kwargs)
        self.dim_X = dim_X
        self.dim_Y = dim_Y

        self.linear_transform = Linear(
                name=self.name + '_linear', input_dim=dim_Y,
                output_dim=dim_X, weights_init=self.weights_init,
                biases_init=self.biases_init, use_bias=self.use_bias)

        self.children = [self.linear_transform]

    @application(inputs=['Y'], outputs=['X_expected'])
    def sample_expected(self, Y):
        return tensor.nnet.sigmoid(self.linear_transform.apply(Y))

    @application(inputs=['Y'], outputs=['X', 'log_prob'])
    def sample(self, Y):
        prob_X = self.sample_expected(Y)
        U = self.theano_rng.uniform(size=prob_X.shape, nstreams=N_STREAMS)
        X = tensor.cast(U <= prob_X, floatX)
        return X, self.log_prob(X, Y)

    @application(inputs=['X', 'Y'], outputs=['log_prob'])
    def log_prob(self, X, Y):
        prob_X = self.sample_expected(Y)
        log_prob = X*tensor.log(prob_X) + (1.-X)*tensor.log(1-prob_X)
        return log_prob.sum(axis=1)

#-----------------------------------------------------------------------------


class GaussianTopLayer(Initializable, ProbabilisticTopLayer):
    @lazy
    def __init__(self, dim_X, biases_init, **kwargs):
        super(GaussianTopLayer, self).__init__(**kwargs)
        self.dim_X = dim_X
        self.biases_init = biases_init

    def _allocate(self):
        b = shared_floatx_zeros((self.dim_X,), name='b')
        add_role(b, BIASES)
        self.parameters = [b]
        
    def _initialize(self):
        b, = self.parameters
        self.biases_init.initialize(b, self.rng)

    @application(inputs=[], outputs=['mean', 'log_sigma'])
    def sample_expected(self, n_samples):
        b, = self.parameters
        mean      = tensor.zeros((n_samples, self.dim_X))
        log_sigma = tensor.zeros((n_samples, self.dim_X)) + b
        return mean, log_sigma

    @application(outputs=['X', 'log_prob'])
    def sample(self, n_samples):
        mean, log_sigma = self.sample_expected(n_samples)

        # Sample from mean-zeros std.-one Gaussian
        U = self.theano_rng.normal(
                    size=(n_samples, self.dim_X),
                    avg=0., std=1.)
        # ... and scale/translate samples
        X = mean + tensor.exp(log_sigma) * U

        return X, self.log_prob(X)

    @application(inputs='X', outputs='log_prob')
    def log_prob(self, X):
        mean, log_sigma = self.sample_expected(X.shape[0])

        # Calculate multivariate diagonal Gaussian
        log_prob =  -0.5*tensor.log(2*numpy.pi) - log_sigma -0.5*(X-mean)**2 / tensor.exp(2*log_sigma)

        return log_prob.sum(axis=1)


class GaussianLayer(Initializable, ProbabilisticLayer):
    @lazy
    def __init__(self, dim_X, dim_Y, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)
        self.dim_X = dim_X
        self.dim_Y = dim_Y
        self.dim_H = (dim_X+dim_Y) // 2

        self.linear_transform = Linear(
                name=self.name + '_linear', input_dim=dim_Y,
                output_dim=self.dim_H, weights_init=self.weights_init,
                biases_init=self.biases_init, use_bias=self.use_bias)

        self.children = [self.linear_transform]
        

    def _allocate(self):
        super(GaussianLayer, self)._allocate()

        dim_X, dim_Y, dim_H = self.dim_X, self.dim_Y, self.dim_H

        W_mean = shared_floatx_zeros((dim_H, dim_X), name='W_mean')
        W_ls   = shared_floatx_zeros((dim_H, dim_X), name='W_ls')
        add_role(W_mean, WEIGHTS)
        add_role(W_ls, WEIGHTS)

        b_mean = shared_floatx_zeros((dim_X,), name='b_mean')
        b_ls   = shared_floatx_zeros((dim_X,), name='b_ls')
        add_role(b_mean, BIASES)
        add_role(b_ls, BIASES)

        self.parameters = [W_mean, W_ls, b_mean, b_ls]
        
    def _initialize(self):
        super(GaussianLayer, self)._initialize()

        W_mean, W_ls, b_mean, b_ls = self.parameters

        self.weights_init.initialize(W_mean, self.rng)
        self.weights_init.initialize(W_ls, self.rng)
        self.biases_init.initialize(b_mean, self.rng)
        self.biases_init.initialize(b_ls, self.rng)

    @application(inputs=['Y'], outputs=['mean', 'log_sigma'])
    def sample_expected(self, Y):
        W_mean, W_ls, b_mean, b_ls = self.parameters

        a = tensor.tanh(self.linear_transform.apply(Y))
        mean      = tensor.dot(a, W_mean) + b_mean
        #log_sigma = tensor.dot(a, W_ls) + b_ls
        log_sigma = tensor.log(0.1)

        return mean, log_sigma

    @application(inputs=['Y'], outputs=['X', 'log_prob'])
    def sample(self, Y):
        mean, log_sigma = self.sample_expected(Y)

        # Sample from mean-zeros std.-one Gaussian
        U = self.theano_rng.normal(
                    size=mean.shape, 
                    avg=0., std=1.)
        # ... and scale/translate samples
        X = mean + tensor.exp(log_sigma) * U

        return X, self.log_prob(X, Y)

    @application(inputs=['X', 'Y'], outputs=['log_prob'])
    def log_prob(self, X, Y):
        mean, log_sigma = self.sample_expected(Y)

        # Calculate multivariate diagonal Gaussian
        log_prob =  -0.5*tensor.log(2*numpy.pi) - log_sigma -0.5*(X-mean)**2 / tensor.exp(2*log_sigma)

        return log_prob.sum(axis=1)

