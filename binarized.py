#!/bin/python3

"""Provides an implementation of a binarized neural network using theano."""

##### Importing Modules #####
### Builtin Modules ###
import abc
### Other Modules ###
import theano
### Package Modules ###
import neuralnet
### Import Specific Functions ###
from abc import ABCMeta, abstractmethod
from theano import function
from theano import tensor

##### Classes #####
class BinarizedLayer(neuralnet.Layer, metaclass = ABCMeta):
    @abstractmethod
    def __init__(self, w_values: "matrix of weight values"):
        self.id = next(Layer.id)
        self._biases_acc = theano.shared(value =
        self._weights_acc = theano.shared(w_values, name = "W", borrow = True)
        
