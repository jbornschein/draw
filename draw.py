from __future__ import division, print_function

import logging
import theano
import theano.tensor as T

from theano import tensor

from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.bricks import Random, MLP, Linear, Tanh, Softmax, Sigmoid, Initializable


class RNN(Initializable):
    def __init__(self, state_dim, input_dim, **kwargs):
        super(RNN, self).__init__(**kwargs)

        transform_dim = state_dim + input_dim

        self.transform = Linear(
                name=self.name+"_transform",
                input_dim=transform_dim, output_dim=state_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)
 
        self.children = [self.transform]
        
    #@application(inputs=['old_state', 'rnn_input'], outputs=['new_state'])
    def apply(self, state, new_input):
        t = T.concatenate([state, new_input], axis=1)
        return T.tanh(self.transform.apply(t))

#-----------------------------------------------------------------------------

class Qsampler(Initializable, Random):
    def __init__(self, input_dim, output_dim, hidden_dim=None, **kwargs):
        super(Qsampler, self).__init__(**kwargs)

        if hidden_dim is None:
            hidden_dim = (input_dim+output_dim) // 2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.h_transform = Linear(
                name=self.name+'_h',
                input_dim=input_dim, output_dim=hidden_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)
        self.mean_transform = Linear(
                name=self.name+'_mean',
                input_dim=hidden_dim, output_dim=output_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)
        self.ls_transform = Linear(
                name=self.name+'_log_sigma',
                input_dim=hidden_dim, output_dim=output_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.h_transform, 
                         self.mean_transform,
                         self.ls_transform]
    
    #@application(inputs=['x'], outputs=['mean', 'log_sigma', 'z'])
    def apply(self, x):
        h = T.tanh(self.h_transform.apply(x))
        mean = self.mean_transform.apply(h)
        log_sigma = self.mean_transform.apply(h)

        # Sample from mean-zeros std.-one Gaussian
        U = self.theano_rng.normal(
                    size=mean.shape, 
                    avg=0., std=1.)

        # ... and scale/translate samples
        z = mean + tensor.exp(log_sigma) * U

        return mean, log_sigma, z

#-----------------------------------------------------------------------------

class Reader(Initializable):
    def __init__(self, x_dim, dec_dim, **kwargs):
        super(Reader, self).__init__(name="reader", **kwargs)

        self.x_dim = x_dim
        self.dec_dim = dec_dim
        self.output_dim = 2*x_dim
            
    @application(inputs=['x', 'x_hat', 'h_dec'], outputs=['r'])
    def apply(self, x, x_hat, h_dec):
        return T.concatenate([x, x_hat], axis=1)


class Writer(Initializable):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Writer, self).__init__(name="writer", **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.transform = Linear(
                name=self.name+'_transform',
                input_dim=input_dim, output_dim=output_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.transform]

    @application(inputs=['h'], outputs=['c_update'])
    def apply(self, h):
        return self.transform.apply(h)


