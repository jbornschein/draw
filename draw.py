#!/usr/bin/env python

from __future__ import division, print_function

import logging

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import ipdb
import theano
import theano.tensor as T

from argparse import ArgumentParser
from collections import OrderedDict

from theano import tensor


from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.datasets.mnist import MNIST, BinarizedMNIST

from blocks.algorithms import GradientDescent, Momentum
from blocks.initialization import Uniform, IsotropicGaussian, Constant, Orthogonal 
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import SerializeMainLoop
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop

from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.bricks import Random, MLP, Linear, Tanh, Softmax, Sigmoid, Initializable
from blocks.bricks.cost import BinaryCrossEntropy

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
        t = T.concatenate([state, new_input])
        return T.tanh(self.transform.apply(t))


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


class Reader(Initializable):
    def __init__(self, x_dim, dec_dim, **kwargs):
        super(Reader, self).__init__(name="reader", **kwargs)

        self.x_dim = x_dim
        self.dec_dim = dec_dim
        self.output_dim = 2*x_dim
            
    @application(inputs=['x', 'x_hat', 'h_dec'], outputs=['r'])
    def apply(self, x, x_hat, h_dec):
        return T.concatenate([x, x_hat])


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
def main(name, epochs, batch_size, learning_rate, n_iter ):
    """ Run a Reweighted Wake Sleep experiment
    """
    if name is None:
        name = "bla"

    #------------------------------------------------------------

    x_dim = 28*28
    read_dim = 2*x_dim
    enc_dim = 100
    dec_dim = 100
    z_dim = 10
    

    prior_mu = T.zeros([z_dim])
    prior_log_sigma = T.zeros([z_dim])

    reader = Reader(x_dim=x_dim, dec_dim=dec_dim)
    writer = Writer(input_dim=dec_dim, output_dim=x_dim)
    encoder = RNN(name="RNN_enc", state_dim=enc_dim, input_dim=(read_dim+dec_dim))
    decoder = RNN(name="RNN_dec", state_dim=dec_dim, input_dim=z_dim)
    q_sampler = Qsampler(input_dim=enc_dim, output_dim=z_dim)
        
    #------------------------------------------------------------------------
    x = tensor.matrix('features')

    # This is one iteration 
    def one_iteration(c, h_enc, z_mu, z_log_sigma, z, h_dec, x):
        x_hat = x-T.nnet.sigmoid(c)
        r = reader.apply(x, x_hat, h_dec)
        h_enc = encoder.apply(h_enc, T.concatenate([r, h_dec]))
        mu, log_sigma, z = q_sampler.apply(h_enc)
        h_dec = decoder.apply(h_dec, z)
        c = c + writer.apply(h_dec)
        return c, h_enc, mu, log_sigma, z, h_dec

    outputs_info = [
            T.zeros_like(x),                  # c
            T.zeros([batch_size, enc_dim]),   # h_enc
            T.zeros([batch_size, z_dim]),     # z_mean
            T.zeros([batch_size, z_dim]),     # z_log_sigma
            T.zeros([batch_size, z_dim]),     # z
            T.zeros([batch_size, dec_dim]),   # h_dec
        ]
    
    outputs, updates = theano.scan(fn=one_iteration, 
                            sequences=[],
                            outputs_info=outputs_info,
                            non_sequences=[x],
                            n_steps=n_iter)

    c, h_enc, z_mean, z_log_sigma, z, h_dec = outputs

    kl_term = (
        prior_log_sigma - z_log_sigma
        + 0.5 * (
            tensor.exp(2 * z_log_sigma) + (z_mean - prior_mu) ** 2
        ) / tensor.exp(2 * prior_log_sigma)
        - 0.5
    ).sum(axis=2).sum(axis=1)
    kl_term.name = "kl_term"
    
    x_hat = T.nnet.sigmoid(c[-1,:,:])
    recons_term = BinaryCrossEntropy().apply(x, x_hat)
    recons_term.name = "recons_term"

    cost = -(recons_term - kl_term).mean()
    cost.name = "nll_bound"

    #------------------------------------------------------------
    #cg = ComputationGraph([cost])
    #for W in VariableFilter(roles=[WEIGHTS])(cg.variables):
    #    cost += 0.00005 * (W**2).sum()
    #cost.name = 'cost'

    algorithm = GradientDescent(
        cost=cost, 
        #step_rule=RMSProp(learning_rate),
        step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
    )

    #------------------------------------------------------------

    #mnist_train = BinarizedMNIST("train", sources=['features'])
    #mnist_test = BinarizedMNIST("test", sources=['features'])
    mnist_train = MNIST("train", binary=True, sources=['features'])
    mnist_test = MNIST("test", binary=True, sources=['features'])

    main_loop = MainLoop(
        model=None,
        data_stream=DataStream(mnist_train,
                        iteration_scheme=SequentialScheme(
                        mnist_train.num_examples, batch_size)),
        algorithm=algorithm,
        extensions=[
            Timing(),
            ProgressBar(),
            FinishAfter(after_n_epochs=epochs),
            DataStreamMonitoring(
                test_costs,
                DataStream(mnist_test,
                    iteration_scheme=SequentialScheme(
                    mnist_test.num_examples, batch_size)),
                    prefix="test"),
            TrainingDataMonitoring(
                [train_cost],   #+[aggregation.mean(algorithm.total_gradient_norm)],
                prefix="train",
                after_every_epoch=True),
            SerializeMainLoop(name+".pkl"),
                Plot(
                    name,
                    channels=[
                        ["train_"+train_cost.name]+["test_%s"%c.name for c in test_costs],
                    ]),
            Printing()])
    main_loop.run()

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                default=None, help="Name for this experiment")
    parser.add_argument("--epochs", type=int, dest="epochs",
                default=100, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=100, help="Size of each mini-batch")
    parser.add_argument("--niter", type=int, dest="n_iter",
                default=10, help="No. of iterations")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=3e-5, help="Learning rate")
    args = parser.parse_args()

    main(**vars(args))

