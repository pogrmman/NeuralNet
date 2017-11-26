#!/bin/python3

"""Provides an implementation of a binarized neural network using theano."""

##### Importing Modules #####
### Builtin Modules ###
import abc
### Other Modules ###
import theano
import numpy
### Package Modules ###
import neuralnet
### Import Specific Functions ###
from abc import ABCMeta, abstractmethod
from theano import function
from theano import tensor

##### Classes #####
#### Base Class ####
### Base Class for Binarized Layers ###
class BinarizedLayer(neuralnet.BasicLayer, metaclass = ABCMeta):
    @classmethod
    def binarize(matrix):
        # Figure out how to binarize theano matrix
        pass
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def make_output(self, input):
        self.output = self.function(tensor.dot(input,
                                           BinarizedLayer.binarize(self.weights)) +
                                           BinarizedLayer.binarize(self.biases))
#### Layer Types ####
## Hyperbolic Tangent Layer ##
class BinTanh(BinarizedLayer):
    """Layer with hyperbolic tangent activation.
    
    Provides the following method:
    function -- tanh activation function
    """
    def __init__(self, rng: "random number generator",
                       inputs: "integer",
                       outputs: "integer"):
        """Set activation function and use parent's __init__ method to complete
           initialization process.
           
        Usage:
        __init__(rng, inputs, outputs)
        
        Arguments:
        rng -- A numpy RandomState.
        inputs -- The number of inputs to this layer.
        outputs -- The number of neurons in this layer.
        """
        self.function = tensor.tanh
        super().__init__(rng, inputs, outputs, init_type = "glorot_tanh")
        
## ReLU Layer ##        
class BinReLU(BinarizedLayer):
    """Layer with rectified linear activation.
    
    Provides the following method:
    function -- relu activation function
    """
    def __init__(self, rng: "random number generator",
                       inputs: "integer",
                       outputs: "integer"):
        """Set activation function and use parent's __init__ method to complete
           initialization process.
           
        Usage:
        __init__(rng, inputs, outputs)
        
        Arguments:
        rng -- A numpy RandomState.
        inputs -- The number of inputs to this layer.
        outputs -- The number of neurons in this layer.
        """
        self.function = tensor.nnet.relu
        super().__init__(rng, inputs, outputs, init_type = "he")
        
## Softplus Layer ##
class BinSoftPlus(BinarizedLayer):
    """Layer with softplus activation.
    
    Provides the following method:
    function -- softplus activation function
    """
    def __init__(self, rng: "random number generator",
                       inputs: "integer",
                       outputs: "integer"):
        """Set activation function and use parent's __init__ method to complete
           initialization process.
           
        Usage:
        __init__(rng, inputs, outputs)
        
        Arguments:
        rng -- A numpy RandomState.
        inputs -- The number of inputs to this layer.
        outputs -- The number of neurons in this layer.
        """
        self.function = tensor.nnet.softplus
        super().__init__(rng, inputs, outputs, init_type = "he")

## Sigmoid Layer ##        
class BinSigmoid(BinarizedLayer):
    """Layer with sigmoid activation.
    
    Provides the following method:
    function -- sigmoid activation function
    """
    def __init__(self, rng: "random number generator",
                       inputs: "integer",
                       outputs: "integer"):
        """Set activation function and use parent's __init__ method to complete
           initialization process.
           
        Usage:
        __init__(rng, inputs, outputs)
        
        Arguments:
        rng -- A numpy RandomState.
        inputs -- The number of inputs to this layer.
        outputs -- The number of neurons in this layer.
        """
        self.function = tensor.nnet.sigmoid
        super().__init__(rng, inputs, outputs)
