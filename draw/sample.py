#!/usr/bin/env python 

from __future__ import print_function, division

import logging
import theano
import theano.tensor as T
import cPickle as pickle

import numpy as np

from PIL import Image

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

def scale_norm(arr):
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    return scale * arr

def img_grid(arr, global_scale=True):
    N, height, width = arr.shape

    rows = int(np.sqrt(N))
    cols = int(np.sqrt(N))

    if rows*cols < N:
        cols = cols + 1

    if rows*cols < N:
        rows = rows + 1

    total_height = rows * height
    total_width  = cols * width

    if global_scale:
        arr = scale_norm(arr)

    I = np.zeros((total_height, total_width))

    for i in xrange(N):
        r = i // cols
        c = i % cols

        if global_scale:
            this = arr[i]
        else:
            this = scale_norm(arr)

        offset_y, offset_x = r*height, c*width
        I[offset_y:(offset_y+height), offset_x:(offset_x+width)] = this

    return Image.fromarray(I, mode='L')


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model_file", help="filename of a pickled DRAW model")
    args = parser.parse_args()

    logging.info("Loading file %s..." % args.model_file)
    with open(args.model_file, "rb") as f:
        main = pickle.load(f)

    model = main.model
    draw = model.get_top_bricks()[0]

    #------------------------------------------------------------
    logging.info("Compiling sample function...")

    n_samples = T.iscalar("n_samples")
    samples = draw.sample(n_samples)

    do_sample = theano.function([n_samples], outputs=samplesm, allow_input_downcast=True)

    #------------------------------------------------------------
    logging.info("Sampling and saving images...")
    
    samples = do_sample(100)
    #samples = np.random.normal(size=(16, 100, 28*28))

    n_iter, N, D = samples.shape

    samples = samples.reshape( (n_iter, N, 28, 28) ) 
    
    for i in xrange(n_iter):
        img = img_grid(samples[i,:,:,:])
        img.save("samples-%03d.png" % i)
        

