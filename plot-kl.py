#!/usr/bin/env python 

from __future__ import division, print_function

import logging
import argparse
import numpy as np
import pylab
import matplotlib as mpl
import cPickle as pickle

from pandas import DataFrame

from mpl_toolkits.mplot3d import Axes3D

from blocks.main_loop import MainLoop
from blocks.log.log import TrainingLogBase


FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model_file", help="filename of a pickled DRAW log")
    args = parser.parse_args()

    logging.info("Loading file %s..." % args.model_file)
    with open(args.model_file, "rb") as f:
        p = pickle.load(f)

    if isinstance(p, MainLoop):
        log = p.log
    elif isinstance(p, TrainingLogBase):
        log = p
    else: 
        print("Don't know how to handle unpickled %s" % type(p))
        exit(1)

    df = DataFrame.from_dict(log, orient='index')
    #df = df.iloc[[0]+log.status._epoch_ends]
    
    cols = ["train_kl_term_%d" % i for i in range(64)]
    cols = filter(lambda col: col in df.columns, cols)

    kl = df[cols]
    kl = np.asarray(kl)
    kl = kl[1:,:]

    print(kl[0,:])
    print(kl[:,0])

    X = np.arange(kl.shape[0])
    Y = np.arange(kl.shape[1])
    X, Y = np.meshgrid(X, Y)

    fig = pylab.figure("KL divergence")
    ax = fig.gca(projection='3d')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Iterations")
    ax.set_zlabel("KL")
    ax.plot_surface(X, Y, kl.T, rstride=5, cstride=5, cmap=mpl.cm.cool, shade=True)
    fig.show()

    pylab.show(block=True)
