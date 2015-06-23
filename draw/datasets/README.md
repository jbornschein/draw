
Sketch
------

[Eitz et al](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) asked non-expert humans to sketch objects of a given category and gather 20,000 unique sketches evenly distributed over 250 object categories.

    @article{eitz2012hdhso,
        author={Eitz, Mathias and Hays, James and Alexa, Marc},
        title={How Do Humans Sketch Objects?},
        journal={ACM Trans. Graph. (Proc. SIGGRAPH)},
        year={2012},
        volume={31},
        number={4},
        pages = {44:1--44:10}
    }

An example of the original sketches:

![original sketches](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/teaser_siggraph.jpg)

The original sketch png files are converted to [fuel's](http://fuel.readthedocs.org/en/latest/h5py_dataset.html)
HDF5 [file](https://s3.amazonaws.com/udidraw/sketch.hdf5) with 
[Sketch.ipynb](http://nbviewer.ipython.org/github/udibr/draw/blob/master/datasets/Sketch.ipynb).

The files created should be placed in the directory `sketch` in fuel data directory

To learn and generate images do

    cd ../draw
    python train-draw.py --name=sketch --sz=56 --attention=2,5 --niter=64 --lr=3e-4 --epochs=100

This is the best result I got so far

 ![generated sketches](https://s3.amazonaws.com/udidraw/sketch-r2-w5-t64-enc256-dec256-z100-lr34.gif)

Not so good... after 100 epochs train_nll_bound: 510.7 test_nll_bound: 542.5, after 200 train was 467.6. after 300 test_nll_bound: 456.5

For attention=4,10 after 100 epochs train_nll_bound: 524.8 test_nll_bound: 536.2 after 200 train_nll_bound: 469.9 test_nll_bound: 470.6
and this is the best result I got so far

 ![generated sketches](https://s3.amazonaws.com/udidraw/sketch-r4-w10-t100-enc256-dec256-z100-lr34.gif)
