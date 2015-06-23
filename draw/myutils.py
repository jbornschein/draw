
from __future__ import division

from abc import ABCMeta, abstractmethod

import ipdb
import numpy
import six
import theano

from collections import OrderedDict

from theano import tensor
from blocks.initialization import NdarrayInitialization, Uniform


def merge_gradients(*gradient_list):
    """Take and merge multiple ordered dicts 
    """
    merged = OrderedDict()
    for gradients in gradient_list:
        assert isinstance(gradients, (dict, OrderedDict))
        for key, val in gradients.items():
            if merged.has_key(key):
                merged[key] = merged[key] + val
            else:       
                merged[key] = val
    return merged

#-----------------------------------------------------------------------------


class ShapeDependentInitialization(NdarrayInitialization):
    """Initialize 

    Parameters
    ----------
    weights_init : :class:`NdarrayInitialization` instance
        The unscaled initialization scheme to initialize the weights with.
    """
    def __init__(self, weights_init):
        super(ShapeDependentInitialization, self).__init__()
        self.weights_init = weights_init

    def generate(self, rng, shape):
        weights = self.weights_init.generate(rng, shape)
        scale = self.scale_func(*shape)
        return scale*weights

    # TODO: Abstract
    def scale_func(self, *shape):
        pass


class TanhInitialization(ShapeDependentInitialization):
    """Normalized initialization for tanh MLPs. 

    This class initializes parameters by drawing from the uniform 
    distribution   with the interval 

        [- sqrt(6)/sqrt(dim_in+dim_out)  .. sqrt(6)/sqrt(dim_in+dim_out)]
    """
    def __init__(self):
        super(TanhInitialization, self).__init__(Uniform(mean=0., width=2.))

    def scale_func(self, dim_in, dim_out):
        return numpy.sqrt(6)/numpy.sqrt(dim_in+dim_out)
