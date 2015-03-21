
Reimplementation of the DRAW network architecture
=================================================

This repository contains a reimplementation of the Deep Recurrent Attentive
Writer (DRAW) network architecture introduced by K. Gregor, I. Danihelka,
A. Graves and D. Wierstra. The original paper can be found at

  http://arxiv.org/pdf/1502.04623


Dependencies
------------
 * [Theano](https://github.com/theano/Theano)
 * [Blocks](https://github.com/bartvm/blocks)
 * [Fuel](https://github.com/bartvm/fuel)


With attention
--------------

To train a model with a 2x2 read and a 5x5 write attention window run

  ./train-draw --attention=2,5 --niter=64 --lr=3e-4 --epochs=100 

It will take about 1 day to train and will create samples similar to 

 ![Samples-r2-w5-t64](doc/mnist-r2-w5-t64-enc256-dec256-z100-lr34.gif)

with the KL divergence plottet over inference iterations and epochs

 ![KL-Divergenc](doc/kl_divergence.png)


Note
----
Work in progress
