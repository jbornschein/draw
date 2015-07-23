#!/usr/bin/env python

from __future__ import division, print_function

import logging

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import theano
import theano.tensor as T
import ipdb
import fuel

from argparse import ArgumentParser
from collections import OrderedDict
from theano import tensor

from fuel.streams import DataStream, ForceFloatX
from fuel.schemes import SequentialScheme
from fuel.datasets.binarized_mnist import BinarizedMNIST

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam, RemoveNotFinite
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal 
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import WEIGHTS, BIASES, PARAMETER
from blocks.model import Model
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

fuel.config.floatX = theano.config.floatX


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

    reader = ZoomableAttentionWindow(img_height, img_width,  read_N)
    writer = ZoomableAttentionWindow(img_height, img_width, write_N)

    # Parameterize the attention reader and writer
    mlpr = MLP(activations=[Tanh(), Identity()], 
                dims=[x_dim, 50, 5], 
                name="RMLP",
                **inits)
    mlpw = MLP(activations=[Tanh(), Identity()],
                dims=[x_dim, 50, 5],
                name="WMLP",
                **inits)

    # MLP between the reader and writer
    mlp = MLP(activations=[Tanh(), Identity()],
                dims=[read_N**2, 300, write_N**2],
                name="MLP",
                **inits)

    for brick in [mlpr, mlpw, mlp]:
        brick.allocate()
        brick.initialize()

    #------------------------------------------------------------------------
    x = tensor.matrix('features')

    hr = mlpr.apply(x)
    hw = mlpw.apply(x)

    center_y, center_x, delta, sigma, gamma = reader.nn2att(hr)
    r = reader.read(x, center_y, center_x, delta, sigma)

    h = mlp.apply(r)

    center_y, center_x, delta, sigma, gamma = writer.nn2att(hw)
    c = writer.write(h, center_y, center_x, delta, sigma) / gamma
    x_recons = T.nnet.sigmoid(c)

    cost = BinaryCrossEntropy().apply(x, x_recons)
    cost.name = "cost"

    #------------------------------------------------------------
    cg = ComputationGraph([cost])
    params = VariableFilter(roles=[PARAMETER])(cg.variables)

    algorithm = GradientDescent(
        cost=cost, 
        parameters=params,
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

    mnist_train = BinarizedMNIST("train", sources=['features'])
    mnist_test = BinarizedMNIST("test", sources=['features'])
    #mnist_train = MNIST("train", binary=True, sources=['features'])
    #mnist_test = MNIST("test", binary=True, sources=['features'])

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

