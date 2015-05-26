#!/usr/bin/env python

from __future__ import division, print_function

import logging
import numpy as np

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import os
import theano
import theano.tensor as T
import fuel
import ipdb

from argparse import ArgumentParser
from collections import OrderedDict
from theano import tensor

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.datasets import H5PYDataset
from fuel.transformers import Flatten

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
import cPickle as pickle

from draw import *

fuel.config.floatX = theano.config.floatX

from sample import generate_samples
from blocks.serialization import secure_pickle_dump
LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"
class MyCheckpoint(Checkpoint):
    def __init__(self, image_size, channels, **kwargs):
        super(MyCheckpoint, self).__init__(**kwargs)
        self.image_size = image_size
        self.channels = channels

    def do(self, callback_name, *args):
        """Pickle the main loop object to the disk.

        If `*args` contain an argument from user, it is treated as
        saving path to be used instead of the one given at the
        construction stage.

        """
        from_main_loop, from_user = self.parse_args(callback_name, args)
        try:
            path = self.path
            if len(from_user):
                path, = from_user
#            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
#            self.main_loop.log.current_row[SAVED_TO] = (
#                already_saved_to + (path,))
#            secure_pickle_dump(self.main_loop, path)
            filenames = self.save_separately_filenames(path)
            for attribute in self.save_separately:
                p = getattr(self.main_loop, attribute)
                if p:
                    secure_pickle_dump(p, filenames[attribute])
                else:
                    print("Empty %s",attribute)
            generate_samples(self.main_loop, self.image_size, self.channels)
        except Exception:
            self.main_loop.log.current_row[SAVED_TO] = None
            raise

#----------------------------------------------------------------------------
def main(name, dataset, epochs, batch_size, learning_rate, 
         attention, n_iter, enc_dim, dec_dim, z_dim, oldmodel, image_size):

    datasource = dataset
    if datasource == 'bmnist':
        if image_size is not None:
            raise Exception('image size for data source %s is pre configured'%datasource)
        image_size = 28
    if datasource == 'sketch':
        if image_size is not None:
            raise Exception('image size for data source %s is pre configured'%datasource)
        image_size = 56
    elif dataset == "cifar10":
        if image_size is not None:
            raise Exception('image size for data source %s is pre configured'%datasource)
        image_size = 32
    else:
        if image_size is None:
            raise Exception('Undefined image size for data source %s'%datasource)

    print("Datasource is " + datasource)

    if datasource == 'bmnist':
        from fuel.datasets.binarized_mnist import BinarizedMNIST
        channels, img_width, img_width = 1, 28, 28
        train_ds = BinarizedMNIST("train", sources=['features'])
        test_ds = BinarizedMNIST("test", sources=['features'])
    elif datasource == 'cifar10':
        from fuel.datasets.cifar10 import CIFAR10
        channels, img_width, img_width = 3, 32, 32
        train_ds = CIFAR10("train", sources=['features'])
        test_ds = CIFAR10("test", sources=['features'])
    elif datasource == 'sketch':
        import sys
        sys.path.append("../datasets")
        from binarized_sketch import BinarizedSketch
        channels, img_width, img_width = 1, 28, 28
        train_ds = BinarizedSketch("train", sources=['features'])
        test_ds = BinarizedSketch("test", sources=['features'])
    else:
        datasource_fname = os.path.join(fuel.config.data_path, datasource , datasource+'.hdf5')
        train_ds = H5PYDataset(datasource_fname, which_set='train', sources=['features'])
        test_ds = H5PYDataset(datasource_fname, which_set='test', sources=['features'])
    train_stream = Flatten(DataStream(train_ds, iteration_scheme=SequentialScheme(train_ds.num_examples, batch_size)))
    test_stream  = Flatten(DataStream(test_ds,  iteration_scheme=SequentialScheme(test_ds.num_examples, batch_size)))


    x_dim = image_size * image_size
    img_height = img_width = image_size

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

        attention_tag = "full"
        # attention_tag = "noatt"

    #----------------------------------------------------------------------

    if name is None:
        name = dataset

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

    lr_str = lr_tag(learning_rate)
    name = "%s-%s-t%d-enc%d-dec%d-z%d-lr%s" % (name, attention_tag, n_iter, enc_dim, dec_dim, z_dim, lr_str)

    print("\nRunning experiment %s" % name)
    print("         learning rate: %g" % learning_rate)
    print("             attention: %s" % attention)
    print("          n_iterations: %d" % n_iter)
    print("     encoder dimension: %d" % enc_dim)
    print("           z dimension: %d" % z_dim)
    print("     decoder dimension: %d" % dec_dim)
    print("            batch size: %d" % batch_size)
    print()

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
            MyCheckpoint(image_size=image_size, channels=channels, path=name+".pkl", before_training=False, after_epoch=True, save_separately=['log', 'model']),
            #Dump(name),
            Plot(name, channels=plot_channels),
            ProgressBar(),
            Printing()])
    if oldmodel is not None:
        print("Initializing parameters with old model %s"%oldmodel)
        with open(oldmodel, "rb") as f:
            oldmodel = pickle.load(f)
            main_loop.model.set_param_values(oldmodel.get_param_values())
        del oldmodel
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
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=100, help="Size of each mini-batch")
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
    parser.add_argument("--oldmodel", type=str,
                help="Use a model pkl file created by a previous run as a starting point for all parameters")
    parser.add_argument("--sz", "--image-size", type=int, dest="image_size",
                help="width and height of each image in dataset")
    args = parser.parse_args()

    main(**vars(args))

