
Reimplementation of the DRAW network architecture
=================================================

This repository contains a reimplementation of the Deep Recurrent Attentive
Writer (DRAW) network architecture introduced by K. Gregor, I. Danihelka,
A. Graves and D. Wierstra. The original paper can be found at

  http://arxiv.org/pdf/1502.04623


Dependencies
------------
 * [Blocks](https://github.com/bartvm/blocks) follow
the [install instructions](http://blocks.readthedocs.org/en/latest/setup.html).
This will install all the other dependencies for you (Theano, Fuel, etc.).
 * [Theano](https://github.com/theano/Theano)
 * [Fuel](https://github.com/bartvm/fuel)
 * [picklable_itertools](https://github.com/dwf/picklable_itertools)
 * [Bokeh](http://bokeh.pydata.org/en/latest/docs/installation.html) 0.8.1+
 * [ipdb](https://pypi.python.org/pypi/ipdb)

Data
----
You need to set the location of your data directory:

    echo "data_path: /home/user/data" >> ~/.fuelrc

You need to download binarized MNIST data:

    export PYLEARN2_DATA_PATH=/home/user/data
    wget https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/datasets/download_binarized_mnist.py
    python download_binarized_mnist.py
    
The [README](./tree/master/datasets) file in the datasets directory has instructions for additional data-sets.

Training with attention
-----------------------
Before training you need to start the bokeh-server

    bokeh-server
or

    boke-server --ip 0.0.0.0

To train a model with a 2x2 read and a 5x5 write attention window run

    ./train-draw --attention=2,5 --niter=64 --lr=3e-4 --epochs=100 

On Amazon g2xlarge it takes more than 40min for Theano's compilation to end and training to start. Once training starts you can track its
[live plotting](http://blocks.readthedocs.org/en/latest/plotting.html).
It will take about 2 days to train the model. After each epoch it will save 3 `pkl` files:
 * a [pickle](https://s3.amazonaws.com/udidraw/mnist-r2-w5-t64-enc256-dec256-z100-lr34.pkl)
of the [enitre main loop](http://blocks.readthedocs.org/en/latest/api/main_loop.html#blocks.main_loop.MainLoop),
 * a [pickle](https://s3.amazonaws.com/udidraw/mnist-r2-w5-t64-enc256-dec256-z100-lr34_log_model.pkl) of the model,
and 
 * a [pickle](https://s3.amazonaws.com/udidraw/mnist-r2-w5-t64-enc256-dec256-z100-lr34_log.pkl)
of the [log](http://blocks.readthedocs.org/en/latest/api/log.html#blocks.log.TrainingLog).

With

    python sample.py [pickle-of-model]
    # this requires ImageMagick to be installed
    convert -delay 5 -loop 0 samples-*.png animaion.gif
you can create samples similar to 

 ![Samples-r2-w5-t64](doc/mnist-r2-w5-t64-enc256-dec256-z100-lr34.gif)

Run 
    
    pyhthon plot-kl.py [pickle-of-log]

to create a visualization of the KL divergence potted over inference iterations and epochs. E.g:

 ![KL-Divergenc](doc/kl_divergence.png)


Testing
-------

Run 

    ./attention.py

to test the attention windowing code. It will open three windows: A window 
displaying the original input image, a window displaying some extracted,
downsampled content (testing the read-operation), and a window showing the
upsampled content (matching the input size) after the write operation.

Note
----
Work in progress
