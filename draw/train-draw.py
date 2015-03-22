#!/usr/bin/env python

from __future__ import division, print_function

import logging
import numpy as np

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import theano
import theano.tensor as T
import fuel
import ipdb

from argparse import ArgumentParser
from collections import OrderedDict
from theano import tensor

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal 
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint, Dump
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.bricks import Identity
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, LSTM

from draw import *

fuel.config.floatX = theano.config.floatX


def get_data(dataset, batch_size):
    if dataset == "bmnist":
        from fuel.datasets.binarized_mnist import BinarizedMNIST

        ds_train = BinarizedMNIST("train", sources=['features'])
        ds_valid = BinarizedMNIST("valid", sources=['features'])
        ds_test = BinarizedMNIST("test", sources=['features'])

        channels, width, height = 1, 28, 28
    elif dataset == "cifar10":
        channels, width, height = 1, 32, 32
    else:
        raise ValueError("Unknown dataset %s" % data)

    streams = [DataStream(
                ds,
                iteration_scheme=SequentialScheme(
                    ds.num_examples, 
                    batch_size
                )) for ds in (ds_train, ds_test)]

    return [width, height]+streams


def float_tag(value):
    """ Convert a float into a short tag-usable string representation. E.g.:
        0.1   -> 11
        0.01  -> 12
        0.001 -> 13
        0.005 -> 53
    """
    exp = np.floor(np.log10(value))
    leading = ("%e"%value)[0]
    return "%s%d" % (leading, -exp)

#----------------------------------------------------------------------------
def main(name, epochs, learning_rate, dataset, 
         attention, n_iter, enc_dim, dec_dim, z_dim):

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

    # Load datasets and create streams
    if dataset == 'bmnist':
        from fuel.datasets.binarized_mnist import BinarizedMNIST

        channels, img_height, img_width = 1, 28, 28

        train_ds = BinarizedMNIST("train", sources=['features'])
        test_ds = BinarizedMNIST("test", sources=['features'])
        #ds_valid = BinarizedMNIST("valid", sources=['features'])
    elif dataset == 'cifar10':
        from fuel.datasets.cifar10 import CIFAR10

        channels, img_height, img_width = 3, 32, 32
            
        train_ds = CIFAR10("train", sources=['features'])
        test_ds = CIFAR10("test", sources=['features'])
    else:
        raise ValueError("Unknown dataset %s" % data)

    batch_size = 100
    train_stream = DataStream(
                    train_ds,
                    iteration_scheme=SequentialScheme(
                        train_ds.num_examples, 
                        batch_size))

    test_stream = DataStream(
                    test_ds,
                    iteration_scheme=SequentialScheme(
                        test_ds.num_examples, 
                        batch_size))

   
    x_dim = channels*img_width*img_height  # a single flattened datapoint

    # Configure attention mechanism
    if attention != "":
        read_N, write_N = attention.split(',')
    
        read_N = int(read_N)
        write_N = int(write_N)
        read_dim = 2*channels*read_N**2

        reader = AttentionReader(x_dim=x_dim, dec_dim=dec_dim,
                                 channels=channels, width=img_width, height=img_height,
                                 N=read_N, **inits)
        writer = AttentionWriter(input_dim=dec_dim, output_dim=x_dim,
                                 channels=channels, width=img_width, height=img_height,
                                 N=write_N, **inits)
        attention_tag = "r%d-w%d" % (read_N, write_N)
    else:
        read_dim = 2*x_dim

        reader = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
        writer = Writer(input_dim=dec_dim, output_dim=x_dim, **inits)

        attention_tag = "noatt"

    #----------------------------------------------------------------------

    encoder_rnn = LSTM(dim=enc_dim, name="RNN_enc", **rnninits)
    decoder_rnn = LSTM(dim=dec_dim, name="RNN_dec", **rnninits)
    encoder_mlp = MLP([Identity()], [(read_dim+dec_dim), 4*enc_dim], name="MLP_enc", **inits)
    decoder_mlp = MLP([Identity()], [             z_dim, 4*dec_dim], name="MLP_dec", **inits)
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

    #----------------------------------------------------------------------

    if name is None:
        name = dataset

    lr_tag = float_tag(learning_rate)
    name = "%s-%s-t%d-enc%d-dec%d-z%d-lr%s" % (name, attention_tag, n_iter, enc_dim, dec_dim, z_dim, lr_tag)

    print("\nRunning experiment %s" % name)
    print("         learning rate: %5.3f" % learning_rate) 
    print("             attention: %s" % attention)
    print("          n_iterations: %d" % n_iter)
    print("     encoder dimension: %d" % enc_dim)
    print("           z dimension: %d" % z_dim)
    print("     decoder dimension: %d" % dec_dim)
    print()

    #------------------------------------------------------------------------
    x = tensor.matrix('features')
    x = x / 255.
    
    x_recons, kl_terms = draw.reconstruct(x)

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
            StepClipping(10.), 
            Adam(learning_rate),
        ])
        #step_rule=RMSProp(learning_rate),
        #step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
    )

    #------------------------------------------------------------------------
    # Setup monitors
    monitors = [cost]
    for t in range(n_iter):
        kl_term_t = kl_terms[t,:].mean()
        kl_term_t.name = "kl_term_%d" % t

        #x_recons_t = T.nnet.sigmoid(c[t,:,:])
        #recons_term_t = BinaryCrossEntropy().apply(x, x_recons_t)
        #recons_term_t = recons_term_t.mean()
        #recons_term_t.name = "recons_term_%d" % t

        monitors +=[kl_term_t]

    train_monitors = monitors[:]
    train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
    train_monitors += [aggregation.mean(algorithm.total_step_norm)]
    # Live plotting...
    plot_channels = [
        ["train_nll_bound", "test_nll_bound"],
        ["train_kl_term_%d" % t for t in range(n_iter)],
        #["train_recons_term_%d" % t for t in range(n_iter)],
        ["train_total_gradient_norm", "train_total_step_norm"]
    ]

    #------------------------------------------------------------

    main_loop = MainLoop(
        model=Model(cost),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            Timing(),
            FinishAfter(after_n_epochs=epochs),
            TrainingDataMonitoring(
                train_monitors, 
                prefix="train",
                after_epoch=True),
#            DataStreamMonitoring(
#                monitors,
#                valid_stream,
##                updates=scan_updates, 
#                prefix="valid"),
            DataStreamMonitoring(
                monitors,
                test_stream,
#                updates=scan_updates, 
                prefix="test"),
            Checkpoint(name+".pkl", after_epoch=True, save_separately=['log', 'model']),
            #Dump(name),
            Plot(name, channels=plot_channels),
            ProgressBar(),
            Printing()])
    main_loop.run()

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                default=None, help="Name for this experiment")
    parser.add_argument("--dataset", "--data", type=str, dest="dataset",
                default="bmnist", help="Dataset to use: [bmnist|cifar10]")
    parser.add_argument("--epochs", type=int, dest="epochs",
                default=100, help="Number of training epochs to do")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=1e-3, help="Learning rate")
    parser.add_argument("--attention", "-a", type=str, default="",
                help="Use attention mechanism (read_window,write_window)")
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

