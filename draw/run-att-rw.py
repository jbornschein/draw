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

from blocks.datasets.streams import DataStream
from blocks.datasets.schemes import SequentialScheme
from blocks.datasets.mnist import MNIST 

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam, RemoveNotFinite
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

from blocks.bricks import Tanh, MLP
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, LSTM

from draw import *
from attention import ZoomableAttentionWindow

#----------------------------------------------------------------------------
def main(name, epochs, batch_size, learning_rate):
    if name is None:
        name = "att-rw" 

    print("\nRunning experiment %s" % name)
    print("         learning rate: %5.3f" % learning_rate) 
    print()


    #------------------------------------------------------------------------

    img_height, img_width = 28, 28
    
    read_N = 12
    write_N = 14

    inits = {
        #'weights_init': Orthogonal(),
        'weights_init': IsotropicGaussian(0.001),
        'biases_init': Constant(0.),
    }
    
    x_dim = img_height * img_width

    reader = ZoomableAttentionWindow(img_height, img_width,  read_N, normalize=True)
    writer = ZoomableAttentionWindow(img_height, img_width, write_N, normalize=True)

    mlp0 = MLP(activations=[Tanh(), Identity()], dims=[    x_dim,  20,          5], **inits)
    mlp1 = MLP(activations=[Tanh(), Identity()], dims=[read_N**2, 200, write_N**2], **inits)
    mlp2 = MLP(activations=[Tanh(), Identity()], dims=[read_N**2, 300,          5], **inits)

    for brick in [mlp0, mlp1, mlp2]:
        brick.allocate()
        brick.initialize()

    #------------------------------------------------------------------------
    x = tensor.matrix('features')

    h0 = mlp0.apply(x)

    center_y  = (h0[:,0] + 1.) / 2.
    center_x  = (h0[:,1] + 1.) / 2.
    delta = T.exp(h0[:,2])
    sigma = T.exp(h0[:,3] / 2.)
    gamma = T.exp(h0[:,4]).dimshuffle(0, 'x')

    r = reader.read(x, center_y, center_x, delta, sigma)

    h1 = mlp1.apply(r)
    h2 = mlp2.apply(r)

    center_y  = (h2[:,0] + 1.) / 2.
    center_x  = (h2[:,1] + 1.) / 2.
    delta = T.exp(h2[:,2])
    sigma = T.exp(h2[:,3] / 2.)
    gamma = T.exp(h2[:,4]).dimshuffle(0, 'x')

    c = writer.write(h1, center_y, center_x, delta, sigma) / gamma
    x_recons = T.nnet.sigmoid(c)

    cost = BinaryCrossEntropy().apply(x, x_recons).mean()
    cost.name = "cost"

    #------------------------------------------------------------
    cg = ComputationGraph([cost])
    params = VariableFilter(roles=[PARAMETER])(cg.variables)

    algorithm = GradientDescent(
        cost=cost, 
        params=params,
        step_rule=CompositeRule([
            RemoveNotFinite(),
            Adam(learning_rate),
            StepClipping(3.), 
        ])
        #step_rule=RMSProp(learning_rate),
        #step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
    )


    #------------------------------------------------------------------------
    # Setup monitors
    monitors = [cost]
    #for v in [center_y, center_x, log_delta, log_sigma, log_gamma]:
    #    v_mean = v.mean()
    #    v_mean.name = v.name
    #    monitors += [v_mean]
    #    monitors += [aggregation.mean(v)]

    train_monitors = monitors[:]
    train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
    train_monitors += [aggregation.mean(algorithm.total_step_norm)]

    # Live plotting...
    plot_channels = [
        ["cost"],
    ]

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
            FinishAfter(after_n_epochs=epochs),
            DataStreamMonitoring(
                monitors,
                DataStream(mnist_test,
                    iteration_scheme=SequentialScheme(
                    mnist_test.num_examples, batch_size)),
                prefix="test"),
            TrainingDataMonitoring(
                train_monitors, 
                prefix="train",
                after_every_epoch=True),
            SerializeMainLoop(name+".pkl"),
            #Plot(name, channels=plot_channels),
            ProgressBar(),
            Printing()])
    main_loop.run()

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                default=None, help="Name for this experiment")
    parser.add_argument("--epochs", type=int, dest="epochs",
                default=25, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=100, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=1e-3, help="Learning rate")
    args = parser.parse_args()

    main(**vars(args))

