
Reimplementation of the DRAW network architecture
=================================================

This repository contains a reimplementation of the Deep Recurrent Attentive
Writer (DRAW) network architecture introduced by K. Gregor, I. Danihelka,
A. Graves and D. Wierstra. The original paper can be found at

  http://arxiv.org/pdf/1502.04623


Dependencies
------------
 * [Blocks](https://github.com/bartvm/blocks) follow
the [install instructions](http://blocks.readthedocs.org/en/latest/setup.html) which will install for you (Theano, Fuel, picklable_itertools)
 * [Theano](https://github.com/theano/Theano)
 * [Fuel](https://github.com/bartvm/fuel)
 * [picklable_itertools](https://github.com/dwf/picklable_itertools)
 * [Bokeh](http://bokeh.pydata.org/en/latest/docs/installation.html) 0.8.1+

Data
----
 * [MNIST](http://blocks.readthedocs.org/en/latest/tutorial.html#training-your-model)
 * [binarized MNIST](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/datasets/download_binarized_mnist.py)

With attention
--------------

To train a model with a 2x2 read and a 5x5 write attention window run

  ./train-draw --attention=2,5 --niter=64 --lr=3e-4 --epochs=100 

You can have [live plotting](http://blocks.readthedocs.org/en/latest/plotting.html) while the process is running.
It will take about 1 day to train and will create samples similar to 

 ![Samples-r2-w5-t64](doc/mnist-r2-w5-t64-enc256-dec256-z100-lr34.gif)

with the KL divergence plottet over inference iterations and epochs

 ![KL-Divergenc](doc/kl_divergence.png)


Testing
-------
  ./attention.py

produce 3 images: original, read: downsampled for input, write: the read upsampled to match input size

Note
----
Work in progress
