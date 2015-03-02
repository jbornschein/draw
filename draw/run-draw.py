#!/usr/bin/env python

from __future__ import division, print_function

import logging
import numpy as np

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import ipdb
import fuel
import theano
import theano.tensor as T

from argparse import ArgumentParser
from collections import OrderedDict
from theano import tensor

from fuel.streams import DataStream, ForceFloatX
from fuel.schemes import SequentialScheme
from fuel.datasets.binarized_mnist import BinarizedMNIST

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal 
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import WEIGHTS, BIASES, PARAMETER
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import SerializeMainLoop
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.bricks import Tanh
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, LSTM

from draw import *


fuel.config.floatX = theano.config.floatX

#----------------------------------------------------------------------------
def main(name, epochs, batch_size, learning_rate, 
         attention, n_iter, enc_dim, dec_dim, z_dim):

     # Learning rate
    def lr_tag(value):
        """ Convert a float into a short tag-usable string representation. E.g.:
            0.1   -> 11
            0.01  -> 12
            0.001 -> 13
            0.005 -> 53
        """
        exp = np.floor(np.log10(value))
        leading = ("%e"%value)[0]
        return "%s%d" % (leading, -exp)

    if name is None:
        tag = "watt" if attention else "woatt"
        lr_str = lr_tag(learning_rate)
        name = "%s-t%d-enc%d-dec%d-z%d-lr%s" % (tag, n_iter, enc_dim, dec_dim, z_dim, lr_str)

    print("\nRunning experiment %s" % name)
    print("         learning rate: %5.3f" % learning_rate) 
    print("             attention: %s" % attention)
    print("          n_iterations: %d" % n_iter)
    print("     encoder dimension: %d" % enc_dim)
    print("           z dimension: %d" % z_dim)
    print("     decoder dimension: %d" % dec_dim)
    print()


    #------------------------------------------------------------------------

    x_dim = 28*28
    img_height, img_width = (28, 28)
    
    rnninits = {
        #'weights_init': Orthogonal(),
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        #'weights_init': Orthogonal(),
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    
    if attention:
        read_N = 4
        write_N = 7
        read_dim = 2*read_N**2

        reader = AttentionReader(x_dim=x_dim, dec_dim=dec_dim,
                                 width=img_width, height=img_height,
                                 N=read_N, **inits)
        writer = AttentionWriter(input_dim=dec_dim, output_dim=x_dim,
                                 width=img_width, height=img_height,
                                 N=read_N, **inits)
    else:
        read_dim = 2*x_dim

        reader = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
        writer = Writer(input_dim=dec_dim, output_dim=x_dim, **inits)

    encoder_rnn = LSTM(dim=enc_dim, name="RNN_enc", **rnninits)
    decoder_rnn = LSTM(dim=dec_dim, name="RNN_dec", **rnninits)
    encoder_mlp = MLP([Tanh()], [(read_dim+dec_dim), 4*enc_dim], name="MLP_enc", **inits)
    decoder_mlp = MLP([Tanh()], [             z_dim, 4*dec_dim], name="MLP_dec", **inits)
    q_sampler = Qsampler(input_dim=enc_dim, output_dim=z_dim, **inits)

    draw = DrawModel(
                n_iter, 
                reader=reader,
                encoder_mlp=encoder_mlp,
                encoder_rnn=encoder_rnn,
                sampler=q_sampler,
                decoder_mlp=decoder_mlp,
                decoder_rnn=decoder_rnn,
                writer=writer)
    draw.initialize()


    #------------------------------------------------------------------------
    x = tensor.matrix('features')
    
    #x_recons = 1. + x
    x_recons, kl_terms = draw.reconstruct(x)
    #x_recons, _, _, _, _ = draw.silly(x, n_steps=10, batch_size=100)
    #x_recons = x_recons[-1,:,:]

    #samples = draw.sample(100) 
    #x_recons = samples[-1, :, :]
    #x_recons = samples[-1, :, :]

    recons_term = BinaryCrossEntropy().apply(x, x_recons)
    recons_term.name = "recons_term"

    cost = recons_term + kl_terms.sum(axis=0).mean()
    cost.name = "nll_bound"

    #------------------------------------------------------------
    cg = ComputationGraph([cost])
    params = VariableFilter(roles=[PARAMETER])(cg.variables)

    algorithm = GradientDescent(
        cost=cost, 
        params=params,
        step_rule=CompositeRule([
            StepClipping(3.), 
            Adam(learning_rate),
        ])
        #step_rule=RMSProp(learning_rate),
        #step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
    )
    #algorithm.add_updates(scan_updates)


    #------------------------------------------------------------------------
    # Setup monitors
    monitors = [cost]
    """
    for t in range(n_iter):
        kl_term_t = kl_terms[t,:].mean()
        kl_term_t.name = "kl_term_%d" % t

        x_recons_t = T.nnet.sigmoid(c[t,:,:])
        recons_term_t = BinaryCrossEntropy().apply(x, x_recons_t)
        recons_term_t = recons_term_t.mean()
        recons_term_t.name = "recons_term_%d" % t

        monitors +=[kl_term_t, recons_term_t]
    """
    train_monitors = monitors[:]
    train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
    train_monitors += [aggregation.mean(algorithm.total_step_norm)]
    # Live plotting...
    plot_channels = [
        ["train_nll_bound", "test_nll_bound"],
        ["train_kl_term_%d" % t for t in range(n_iter)],
        ["train_recons_term_%d" % t for t in range(n_iter)],
        ["train_total_gradient_norm", "train_total_step_norm"]
    ]

    #------------------------------------------------------------

    mnist_train = BinarizedMNIST("train", sources=['features'])
    mnist_test = BinarizedMNIST("test", sources=['features'])

    main_loop = MainLoop(
        model=Model(cost),
        data_stream=ForceFloatX(DataStream(mnist_train,
                        iteration_scheme=SequentialScheme(
                        mnist_train.num_examples, batch_size))),
        algorithm=algorithm,
        extensions=[
            Timing(),
            FinishAfter(after_n_epochs=epochs),
            DataStreamMonitoring(
                monitors,
                ForceFloatX(DataStream(mnist_test,
                    iteration_scheme=SequentialScheme(
                    mnist_test.num_examples, batch_size))),
##                updates=scan_updates, 
                prefix="test"),
            TrainingDataMonitoring(
                train_monitors, 
                prefix="train",
                after_every_epoch=True),
            SerializeMainLoop(name+".pkl"),
            Plot(name, channels=plot_channels),
            ProgressBar(),
            Printing()])
    main_loop.run()

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                default=None, help="Name for this experiment")
    parser.add_argument("--epochs", type=int, dest="epochs",
                default=100, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=100, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=1e-3, help="Learning rate")
    parser.add_argument("--attention", "-a", action="store_true",
                help="Use attention mechanism")
    parser.add_argument("--niter", type=int, dest="n_iter",
                default=10, help="No. of iterations")
    parser.add_argument("--enc-dim", type=int, dest="enc_dim",
                default=256, help="Encoder RNN state dimension")
    parser.add_argument("--dec-dim", type=int, dest="dec_dim",
                default=256, help="Decoder  RNN state dimension")
    parser.add_argument("--z-dim", type=int, dest="z_dim",
                default=100, help="Z-vector dimension")
    args = parser.parse_args()

    main(**vars(args))

