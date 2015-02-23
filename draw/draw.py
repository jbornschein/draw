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
    def apply(self, state, new_input, iterate=False):
        assert not iterate
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


#-----------------------------------------------------------------------------


"""
class DrawModel(Initializable):
    def __init__(self, reader, encoder, encoder_mlp, sampler, decoder, decoder_mlp, writer, **kwargs):
        super(DrawModel, self).__init__(**kwargs)

        self.reader = reader
        self.encoder = encoder
        self.encoder_mlp = encoder_mlp 
        self.sampler = sampler
        self.decoder = decoder
        self.decoder_mlp = encoder_mlp 
        self.writer = writer

        self.children = [self.reader, self.encoder, self.sampler, 
                         self.decoder, self.writer]
    
    @application(inputs=['features'], outputs=['recons'])
    def apply(self, features):

        encoder = self.encoder
        decoder = self.decoder

        batch_size = features.shape[0]

        # This is one iteration 
        def one_iteration(c, h_enc, z_mean, z_log_sigma, z, h_dec, x):
            x_hat = x-T.nnet.sigmoid(c)
            r = self.reader.apply(x, x_hat, h_dec)
            i_enc = self.encoder_mlp.apply(T.concatenate([r, h_dec], axis=1))
            h_enc = self.encoder.apply(states=h_enc, input=i_enc, iterate=False)
            z_mean, z_log_sigma, z = self.q_sampler.apply(h_enc)
            i_dec = self.decoder_mlp.apply(z)
            h_dec = self.decoder.apply(states=h_dec, input=i_dec, iterate=False)
            c = c + self.writer.apply(h_dec)
            return c, h_enc, z_mean, z_log_sigma, z, h_dec

        outputs_info = [
            T.zeros([batch_size, x_dim]),     # c
            T.zeros([batch_size, enc_dim]),   # h_enc
            T.zeros([batch_size, z_dim]),     # z_mean
            T.zeros([batch_size, z_dim]),     # z_log_sigma
            T.zeros([batch_size, z_dim]),     # z
            T.zeros([batch_size, dec_dim]),   # h_dec
        ]
    
        outputs, scan_updates = theano.scan(fn=one_iteration, 
                                            sequences=[],
                                            outputs_info=outputs_info,
                                            non_sequences=[x],
                                            n_steps=n_iter)

        c, h_enc, c_enc, z_mean, z_log_sigma, z, h_dec, c_dec = outputs

        kl_terms = (
            prior_log_sigma - z_log_sigma
            + 0.5 * (
                tensor.exp(2 * z_log_sigma) + (z_mean - prior_mu) ** 2
                ) / tensor.exp(2 * prior_log_sigma)
        - 0.5
    ).sum(axis=-1)
    
    x_recons = T.nnet.sigmoid(c[-1,:,:])
    recons_term = BinaryCrossEntropy().apply(x, x_recons)
    recons_term.name = "recons_term"

    cost = (recons_term + kl_terms.sum(axis=0)).mean()
    cost.name = "nll_bound"


"""
