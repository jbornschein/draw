from __future__ import division, print_function

import sys
sys.path.append("../lib")

import logging
import theano
import theano.tensor as T

from theano import tensor

from blocks.bricks.base import application, lazy
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Random, Initializable, MLP, Linear
from blocks.bricks import Identity, Tanh, Logistic

from attention import ZoomableAttentionWindow
from prob_layers import replicate_batch

#-----------------------------------------------------------------------------

class Qsampler(Initializable, Random):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Qsampler, self).__init__(**kwargs)

        self.prior_mean = 0.
        self.prior_log_sigma = 0.

        self.mean_transform = Linear(
                name=self.name+'_mean',
                input_dim=input_dim, output_dim=output_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.log_sigma_transform = Linear(
                name=self.name+'_log_sigma',
                input_dim=input_dim, output_dim=output_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.mean_transform, self.log_sigma_transform]
    
    def get_dim(self, name):
        if name == 'input':
            return self.mean_transform.get_dim('input')
        elif name == 'output':
            return self.mean_transform.get_dim('output')
        else:
            raise ValueError

    @application(inputs=['x', 'u'], outputs=['z', 'kl_term'])
    def sample(self, x, u):
        """Return a samples and the corresponding KL term

        Parameters
        ----------
        x : 

        Returns
        -------
        z : tensor.matrix
            Samples drawn from Q(z|x) 
        kl : tensor.vector
            KL(Q(z|x) || P_z)
        
        """
        mean = self.mean_transform.apply(x)
        log_sigma = self.log_sigma_transform.apply(x)

        # Sample from mean-zeros std.-one Gaussian
        #u = self.theano_rng.normal(
        #            size=mean.shape, 
        #            avg=0., std=1.)

        # ... and scale/translate samples
        z = mean + tensor.exp(log_sigma) * u

        # Calculate KL
        kl = (
            self.prior_log_sigma - log_sigma
            + 0.5 * (
                tensor.exp(2 * log_sigma) + (mean - self.prior_mean) ** 2
                ) / tensor.exp(2 * self.prior_log_sigma)
            - 0.5
        ).sum(axis=-1)
 
        return z, kl

    #@application(inputs=['n_samples'])
    @application(inputs=['u'], outputs=['z_prior'])
    def sample_from_prior(self, u):
        """Sample z from the prior distribution P_z.

        Parameters
        ----------
        u : tensor.matrix
            gaussian random source 

        Returns
        -------
        z : tensor.matrix
            samples 

        """
        z_dim = self.mean_transform.get_dim('output')
    
        # Sample from mean-zeros std.-one Gaussian
        #u = self.theano_rng.normal(
        #            size=(n_samples, z_dim),
        #            avg=0., std=1.)

        # ... and scale/translate samples
        z = self.prior_mean + tensor.exp(self.prior_log_sigma) * u
        #z.name("z_prior")
    
        return z

#-----------------------------------------------------------------------------


class Reader(Initializable):
    def __init__(self, x_dim, dec_dim, **kwargs):
        super(Reader, self).__init__(name="reader", **kwargs)

        self.x_dim = x_dim
        self.dec_dim = dec_dim
        self.output_dim = 2*x_dim

    def get_dim(self, name):
        if name == 'input':
            return self.dec_dim
        elif name == 'x_dim':
            return self.x_dim
        elif name == 'output':
            return self.output_dim
        else:
            raise ValueError

    @application(inputs=['x', 'x_hat', 'h_dec'], outputs=['r'])
    def apply(self, x, x_hat, h_dec):
        return T.concatenate([x, x_hat], axis=1)

class AttentionReader(Initializable):
    def __init__(self, x_dim, dec_dim, channels, height, width, N, **kwargs):
        super(AttentionReader, self).__init__(name="reader", **kwargs)

        self.img_height = height
        self.img_width = width
        self.N = N
        self.x_dim = x_dim
        self.dec_dim = dec_dim
        self.output_dim = 2*channels*N*N

        self.zoomer = ZoomableAttentionWindow(channels, height, width, N)
        self.readout = MLP(activations=[Identity()], dims=[dec_dim, 5], **kwargs)

        self.children = [self.readout]

    def get_dim(self, name):
        if name == 'input':
            return self.dec_dim
        elif name == 'x_dim':
            return self.x_dim
        elif name == 'output':
            return self.output_dim
        else:
            raise ValueError
            
    @application(inputs=['x', 'x_hat', 'h_dec'], outputs=['r'])
    def apply(self, x, x_hat, h_dec):
        l = self.readout.apply(h_dec)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        w     = gamma * self.zoomer.read(x    , center_y, center_x, delta, sigma)
        w_hat = gamma * self.zoomer.read(x_hat, center_y, center_x, delta, sigma)
        
        return T.concatenate([w, w_hat], axis=1)

#-----------------------------------------------------------------------------

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


class AttentionWriter(Initializable):
    def __init__(self, input_dim, output_dim, channels, width, height, N, **kwargs):
        super(AttentionWriter, self).__init__(name="writer", **kwargs)

        self.channels = channels
        self.img_width = width
        self.img_height = height
        self.N = N
        self.input_dim = input_dim
        self.output_dim = output_dim

        assert output_dim == channels*width*height

        self.zoomer = ZoomableAttentionWindow(channels, height, width, N)
        self.z_trafo = Linear(
                name=self.name+'_ztrafo',
                input_dim=input_dim, output_dim=5, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.w_trafo = Linear(
                name=self.name+'_wtrafo',
                input_dim=input_dim, output_dim=channels*N*N, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.z_trafo, self.w_trafo]

    @application(inputs=['h'], outputs=['c_update'])
    def apply(self, h):
        w = self.w_trafo.apply(h)
        l = self.z_trafo.apply(h)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        c_update = 1./gamma * self.zoomer.write(w, center_y, center_x, delta, sigma)

        return c_update

    @application(inputs=['h'], outputs=['c_update', 'center_y', 'center_x', 'delta'])
    def apply_detailed(self, h):
        w = self.w_trafo.apply(h)
        l = self.z_trafo.apply(h)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        c_update = 1./gamma * self.zoomer.write(w, center_y, center_x, delta, sigma)

        return c_update, center_y, center_x, delta



#-----------------------------------------------------------------------------


class DrawModel(BaseRecurrent, Initializable, Random):
    def __init__(self, n_iter, reader, 
                    encoder_mlp, encoder_rnn, sampler, 
                    decoder_mlp, decoder_rnn, writer, **kwargs):
        super(DrawModel, self).__init__(**kwargs)   
        self.n_iter = n_iter

        self.reader = reader
        self.encoder_mlp = encoder_mlp 
        self.encoder_rnn = encoder_rnn
        self.sampler = sampler
        self.decoder_mlp = decoder_mlp 
        self.decoder_rnn = decoder_rnn
        self.writer = writer

        self.children = [self.reader, self.encoder_mlp, self.encoder_rnn, self.sampler, 
                         self.writer, self.decoder_mlp, self.decoder_rnn]
 
    def get_dim(self, name):
        if name == 'c':
            return self.reader.get_dim('x_dim')
        elif name == 'h_enc':
            return self.encoder_rnn.get_dim('states')
        elif name == 'c_enc':
            return self.encoder_rnn.get_dim('cells')
        elif name in ['z', 'z_mean', 'z_log_sigma']:
            return self.sampler.get_dim('output')
        elif name == 'h_dec':
            return self.decoder_rnn.get_dim('states')
        elif name == 'c_dec':
            return self.decoder_rnn.get_dim('cells')
        elif name == 'kl':
            return 0
        elif name == 'center_y':
            return 0
        elif name == 'center_x':
            return 0
        elif name == 'delta':
            return 0
        else:
            super(DrawModel, self).get_dim(name)

    #------------------------------------------------------------------------

    @recurrent(sequences=['u'], contexts=['x'], 
               states=['c', 'h_enc', 'c_enc', 'z', 'kl', 'h_dec', 'c_dec'],
               outputs=['c', 'h_enc', 'c_enc', 'z', 'kl', 'h_dec', 'c_dec'])
    def apply(self, u, c, h_enc, c_enc, z, kl, h_dec, c_dec, x):
        x_hat = x-T.nnet.sigmoid(c)
        r = self.reader.apply(x, x_hat, h_dec)
        i_enc = self.encoder_mlp.apply(T.concatenate([r, h_dec], axis=1))
        h_enc, c_enc = self.encoder_rnn.apply(states=h_enc, cells=c_enc, inputs=i_enc, iterate=False)
        z, kl = self.sampler.sample(h_enc, u)

        i_dec = self.decoder_mlp.apply(z)
        h_dec, c_dec = self.decoder_rnn.apply(states=h_dec, cells=c_dec, inputs=i_dec, iterate=False)
        c = c + self.writer.apply(h_dec)
        return c, h_enc, c_enc, z, kl, h_dec, c_dec

    @recurrent(sequences=['u'], contexts=[], 
               states=['c', 'h_dec', 'c_dec'],
               outputs=['c', 'h_dec', 'c_dec'])
    def decode(self, u, c, h_dec, c_dec):
        batch_size = c.shape[0]

        z = self.sampler.sample_from_prior(u)
        i_dec = self.decoder_mlp.apply(z)
        h_dec, c_dec = self.decoder_rnn.apply(
                    states=h_dec, cells=c_dec, 
                    inputs=i_dec, iterate=False)
        c = c + self.writer.apply(h_dec)
        return c, h_dec, c_dec

    #------------------------------------------------------------------------

    @application(inputs=['features'], outputs=['recons', 'kl'])
    def reconstruct(self, features):
        batch_size = features.shape[0]
        dim_z = self.get_dim('z')

        # Sample from mean-zeros std.-one Gaussian
        u = self.theano_rng.normal(
                    size=(self.n_iter, batch_size, dim_z),
                    avg=0., std=1.)

        c, h_enc, c_enc, z, kl, h_dec, c_dec = \
            rvals = self.apply(x=features, u=u)

        x_recons = T.nnet.sigmoid(c[-1,:,:])
        x_recons.name = "reconstruction"

        kl.name = "kl"

        return x_recons, kl

    @application(inputs=['n_samples'], outputs=['samples'])
    def sample(self, n_samples):
        """Sample from model.

        Returns 
        -------

        samples : tensor3 (n_samples, n_iter, x_dim)
        """
    
        # Sample from mean-zeros std.-one Gaussian
        u_dim = self.sampler.mean_transform.get_dim('output')
        u = self.theano_rng.normal(
                    size=(self.n_iter, n_samples, u_dim),
                    avg=0., std=1.)

        c, _, _, = self.decode(u)
        #c, _, _, center_y, center_x, delta = self.decode(u)
        return T.nnet.sigmoid(c)
