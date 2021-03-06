#!/bin/python3

"""Provides an implementation of a feedfoward neural network using theano."""

##### Importing Modules #####
### Builtin Modules ###
import itertools
import random
import pickle
import collections
import abc
### Other Modules ###
import theano
import numpy
### Import Specific Functions ###
from abc import ABCMeta, abstractmethod
from theano import function
from theano import tensor

##### Classes #####
### Abstract Base Classes ###
## Layer Base Classes ##
# Base Layer Class #
class Layer(object, metaclass = ABCMeta):
    """Parent class for all layer types.
    
    Provides the following public method:
    make_output -- Build a theano expression for the output of this layer.
    
    Provides the following attributes:
    weights -- The weights of this layer.
    biases -- The biases of this layer.
    id -- A layer id number.
    params -- A list of weights and biases.
    output -- A theano expression for this layer's output. Does not exist until
              make_output is called.
    """
    id = itertools.count()
    
    @abstractmethod
    def __init__(self, w_values: "matrix of weight values"):
        """Finish layer initialization.
        
        Usage:
        __init__(w_values)
        
        Arguments:
        w_values -- A matrix containing the initialized weight values.
        
        This method is only intended to be called to complete initialization of
        children of this class.
        """
        self.id = next(Layer.id)
        self.biases = theano.shared(value = numpy.zeros((self.outputs,),
                                                  dtype = theano.config.floatX),
                                    name = "b",
                                    borrow = True)
        self.weights = theano.shared(w_values, name = "W", borrow = True)
        self.params = [self.weights, self.biases]
        
    def make_output(self, input):
        """Build a theano expression for this layer's output.
        
        Usage:
        make_output(input)
        
        Arguments:
        input -- A theano matrix of the inputs to this layer.
        """
        self.output = self.function(tensor.dot(input, self.weights) + 
                                    self.biases)

# Simple Layers Class #                                    
class BasicLayer(Layer, metaclass = ABCMeta):
    """Parent class for most layer types.
    
    Provides the following attributes:
    outputs -- Number of neurons in layer.
    inputs -- Number of inputs to layer.
    """
    @abstractmethod
    def __init__(self, rng: "random number generator",
                       inputs: "integer",
                       outputs: "integer",
                       init_type: "string" = "glorot"):
        """Initialize the layer.
        
        Usage:
        __init__(rng, inputs, outputs)
        
        Arguments:
        rng -- A numpy RandomState.
        inputs -- The number of inputs to this layer.
        outputs -- The number of neurons in this layer.
        init_type -- The type of initialization to use.
        
        May raise a not implemented error if the init_type is not supported.
        This method is only intended to be called by children of this class
        during their initialization.
        """
        if init_type == "glorot": # Glorot, Bengio 2010 
            init_limit = numpy.sqrt(6. / (inputs + outputs)) 
            w_values = numpy.asarray(rng.uniform(low = -init_limit,
                                                 high = init_limit,
                                                 size = (inputs, outputs)),
                                     dtype = theano.config.floatX)
        elif init_type == "glorot_tanh": # Glorot, Bengio 2010 
            init_limit = 4 * numpy.sqrt(6. / (inputs + outputs)) 
            w_values = numpy.asarray(rng.uniform(low = -init_limit,
                                                 high = init_limit,
                                                 size = (inputs, outputs)),
                                     dtype = theano.config.floatX)
        elif init_type == "he": # He, et al 2015
            init_std_dev = numpy.sqrt(2 / outputs)
            w_values = numpy.asarray(rng.normal(loc = 0,
                                                scale = init_std_dev,
                                                size = (inputs, outputs)),
                                     dtype = theano.config.floatX)
        else:
            raise NotImplementedError("The initilization type " + init_type +
                                      " is not supported")
        self.outputs = outputs
        self.inputs = inputs
        super().__init__(w_values)
    
### Network Class ###
class Network(object):
    """Create a feedforward neural network
    
    Provides the following public methods:
    train -- Trains the neural network.
    forwardprop -- Forwardpropagate data through the network.
    backprop -- Backpropagate through the network and update weights and biases.    

    Provides the following attributes:
    layers -- A list of Layer objects.
    cost -- A theano expression for the crossentropy cost of a training example.
    params -- A list of all the network's weights and biases.
    """
    def __init__(self, net_data: "list of tuples", rate: "float", 
                       reg_coeff: "float" = 0, momentum_coeff: "float" = 0,
                       cost_type: "string"  = "categorical crossentropy",
                       seed: "integer" = 100, early_stop: "boolean" = False):
        """Initialize the neural network
        
        Usage:
        __init__(net_data, rate[, reg_coeff, momentum_coeff, seed, early_stop])
        
        Arguments:
        net_data -- A list of tuples of the form (number of neurons, layer type)
                    that describes the architecture of the network.
                    The first item is the first layer, and so on.
        rate -- The learning rate for the network, a floating point number.
        reg_coeff -- The L2 regularization coefficient. Defaults to 0.
        momentum_coeff -- The coefficient for momentum degradation.
        cost_type -- The type of cost function.
        seed -- A seed for the rng for initialization.
        early_stop -- A boolean controlling whether to use early stopping.
        """
        reset_layer_ids()
        self._data = net_data
        self.layers = []
        self._rng = numpy.random.RandomState(seed)
        for i in range(1, len(self._data)):
            kind = self._data[i][1]
            inputs = self._data[i-1][0]
            outputs = self._data[i][0]
            self.layers.append(self._make_layer(kind, inputs, outputs))
        if type(self.layers[-1]) == Softmax:
            self.make_prediction = self._make_prediction
        self._set_cost(cost_type)
        self._set_train(early_stop)
        self._build_forwardprop()
        self._build_backprop(rate, reg_coeff, momentum_coeff)
        
    def _make_layer(self, kind: "layer type",
                         inputs: "integer", 
                         outputs: "integer") -> "layer":
        """Make a layer
        
        Usage:
        _make_layer(kind, inputs, outputs)
        
        Arguments:
        kind -- A layer type
        inputs -- The number of inputs the layer takes
        outputs -- The number of outputs the layer takes
        
        This method may throw a NotImplementedError if the kind of layer
        specified does not exist.
        
        Not intended to be accessed publicly.
        """
        if kind == "softmax":
            return Softmax(self._rng, inputs, outputs)
        elif kind == "sigmoid":
            return Sigmoid(self._rng, inputs, outputs)
        elif kind == "tanh":
            return Tanh(self._rng, inputs, outputs)
        elif kind == "relu":
            return ReLU(self._rng, inputs, outputs)
        elif kind == "softplus":
            return SoftPlus(self._rng, inputs, outputs)
        else:
            raise NotImplementedError("The layer type " + kind +
                                      " is unimplemented")
    
    def _set_cost(self, type: "cost type"):
        """Set the _costfunc attributes
        
        Usage:
        _set_cost(type)
        
        Arguments:
        type -- the cost function type
        
        This method may throw a NotImplementedError if the kind of cost function
        has not yet been implemented.
        
        This method is used by __init__ to set the cost type appropriately.
        Not intended to be accessed publicly.
        """
        if type == "categorical crossentropy":
            self._costfunc = tensor.nnet.categorical_crossentropy
        elif type == "binary crossentropy":
            self._costfunc = tensor.nnet.binary_crossentropy
        elif type == "quadratic":
            self._costfunc = quadratic_cost
        else:
            raise NotImplementedError("The cost type " + kind +
                                       " is unimplemented.")
                                       
    def _set_train(self, early_stop: "boolean"):
        if early_stop:
            self.train = self._early_stop_train
        else:
            self.train = self._basic_train
            
    def _build_forwardprop(self):
        """Compile a theano function for forwardpropagation
        
        Usage:
        _build_forwardprop()
        
        This method is used by __init__ to create the forwardprop method.
        Not intended to be accessed publicly.
        """
        # Make theano symbols for input and output
        self._inpt = tensor.fmatrix("inpt")
        self._otpt = tensor.fmatrix("otpt")
        self.layers[0].make_output(self._inpt)
        for layer in self.layers:
            if layer.id != 0:
                layer.make_output(self.layers[layer.id - 1].output)
        self._output = self.layers[-1].output
        # Compile forwardprop method
        self.forwardprop = function(inputs = [self._inpt],
                                    outputs = self._output,
                                    allow_input_downcast = True)
        
    def _build_backprop(self, rate: "float", reg_coeff: "float",
                              momentum_coeff: "float" = 0):
        """Compile a theano function for backpropagation
        
        Usage:
        _build_backprop(rate, reg_coeff[, momentum_coeff])
        
        Arguments:
        rate -- The learning rate for the network.
        reg_coeff -- The L2 regularization coefficient.
        momentum_coeff -- Degradatoin coefficient for momentum.
        
        This method is used by __init__ to create the backprop method.
        Not intended to be accessed publicly.
        """
        # L2 regularization expression
        regularize = 0
        for layer in self.layers:
            regularize += abs(layer.weights).sum() ** 2
        self.cost = (tensor.mean(self._costfunc(self._output, self._otpt)) +
                     (reg_coeff * regularize))
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params[0])
            self.params.append(layer.params[1])
        self._gradients = tensor.grad(cost = self.cost, wrt = self.params)
        self._updates = []
        for grad, param in zip(self._gradients, self.params):
            param_update = theano.shared(param.get_value()*0.,
                                         broadcastable = param.broadcastable)
            self._updates.append([param, param - (rate * param_update)])
            self._updates.append([param_update, momentum_coeff * param_update +
                                                (1. - momentum_coeff) * grad])
        # Compile backprop method
        self.backprop = function(inputs = [self._inpt, self._otpt], 
                                 outputs = self.cost,
                                 updates = self._updates,
                                 allow_input_downcast = True)
        # Compile cost method that does not update params
        self.cost_calc = function(inputs = [self._inpt, self._otpt],
                                  outputs = self.cost,
                                  allow_input_downcast = True)

    def _data_generator(self, data: "list of lists",
                              epochs: "integer",
                              minibatch_size: "integer"):
        minibatches_per_epoch = len(data) // minibatch_size
        for i in range(0, epochs):
            for j in range(0, minibatches_per_epoch):
                yield (i, self._make_minibatch(data, minibatch_size))
                
    def _basic_train(self, data: "list of lists", 
                           epochs: "integer",
                           minibatch_size: "integer"):
        """Train the neural network using SGD.
        
        Usage:
        train(data, epochs, minbatch_size)
        
        Arguments:
        data -- A list of training examples of the form 
                [[data], [intended output]].
        epochs -- The number of epochs to train for.
        minbatch_size -- The size of a minibatch.
        
        This method updates the weights and biases of the network using the
        backprop method.
        """
        data_gen = self._data_generator(data, epochs, minibatch_size)
        for epoch, item in data_gen:
            self.backprop(item[0], item[1])

    def _early_stop_train(self, data: "list of lists",
                                epochs: "integer",
                                validation: "list of lists",
                                minibatch_size: "integer",
                                min_epochs = 0,
                                check_every = 0,
                                tolerance = 0.20):
        """Train the neural network with SGD and early stopping.
        
        Usage:
        train(data, epochs, validation, minbatch_size[, min_epochs, check_every, threshold])

        Arguments:
        data -- A list of training examples of the form
                [[data],[intended output]].
        epochs -- The number of epochs to train for.
        validation -- A list of validation examples of the form
                      [[data],[intended output]].
        minbatch_size -- The size of a minibatch.
        min_epochs -- The minimum number of epochs to train for. Defaults to
                      1/5 of epochs.
        check_every -- Check validation cost after this many epochs. Defaults to
                       1/5 of min_epochs.
        threshold -- The tolerance for increases in error as a porportion.
                     Defaults to 0.20.

        This method updates the weights and biases of the network using the
        backprop method. It stops the training early if the performance on
        the validation data decreases too much.
        """
        data_gen = self._data_generator(data, epochs, minibatch_size)
        val_gen = self._data_generator(validation, epochs, minibatch_size)
        if min_epochs == 0:
            min_epochs = epochs // 5
        if check_every == 0:
            check_every = min_epochs // 5
        val_epoch, val_item = next(val_gen)
        cost = numpy.mean(self.cost_calc(val_item[0],val_item[1]))
        print("Epoch 0 -- cost is " + str(round(cost,2)))
        costs = [cost]
        last_check = 0
        for epoch, item in data_gen:
            self.backprop(item[0],item[1])
            if epoch != last_check and epoch % check_every == 0:
                last_check = epoch
                val_epoch, val_item = next(val_gen)
                cost = numpy.mean(self.cost_calc(val_item[0],val_item[1]))
                print("Epoch " + str(epoch) + " -- cost is " + str(round(cost,2)))
                avg = numpy.mean(costs)
                threshold = avg * tolerance
                if epoch > min_epochs and cost > avg + threshold:
                    print("Stopping early!")
                    break
                else:
                    costs = costs[1:]
                    costs.append(cost)

    def _make_minibatch(self, data, minbatch_size):
        items = [random.choice(data) for i in range(0, minbatch_size)]
        inpts = [item[0] for item in items]
        otpts = [item[1] for item in items]
        return (inpts,otpts)
    
    def _make_prediction(self, datum: "list"):
        return numpy.argmax(self.forwardprop([datum]))
    
### BuildNetwork Class ###
class BuildNetwork(Network):
    """Builds a network from a list of layers."""
    def __init__(self, layer_list: "list of Layer objects", rate: "float",
                       reg_coeff: "float" = 0, momentum_coeff: "float" = 0,
                       cost_type: "string" = "categorical crossentropy",
                       early_stop: "boolean" = False):
        """Create a network from a list of layers.
        
        Usage:
        __init__(layer_list, rate[, reg_coeff, cost_type, early_stop])
        
        Arguments:
        layer_list -- A list of Layer objects that have the appropriate number
                      of inputs.
        rate -- The learning rate for the network, a floating point number.
        reg_coeff -- The L2 regularization coefficient, a floating point number.
        momentum_coeff -- The coefficient for momentum degradation.
        cost_type -- The cost function to use, a string.
        early_stop -- A boolean controlling whether to use early stopping.
        
        May raise a type error if the Layer objects don't have the appropriate 
        number of inputs. 
        """
        self.layers = layer_list
        for i in range(0,len(self.layers)):
            self.layers[i].id = i
        # Check to see if these layer objects make a valid network
        if len(self.layers) > 1:
            for i in range(1,len(self.layers)):
                if self.layers[i].inputs != self.layers[i-1].outputs:
                    raise TypeError("The Layer objects specified cannot be " +
                                    "used to create a valid neural network, " +
                                    "as there is a mismatch between the " +
                                    "number of inputs of one layer and the " +
                                    "number of neurons in the previous layer.")
        self._set_cost(cost_type)
        self._set_train(early_stop)
        self._build_forwardprop()
        self._build_backprop(rate, reg_coeff, momentum_coeff)

### EnsembleClassifier Class ###
class EnsembleClassifier(object):
    """Builds an ensemble classifier from trained neural nets.

    Provides the following method:
    make_predicion -- 
    """
    def __init__(self, nets):
        self.nets = nets
    def make_prediction(self, datum):
        preds = [numpy.argmax(net.forwardprop([datum])) for net in self.nets]
        preds = collections.Counter(preds)
        return preds.most_common(1)[0][0]
    
### Layer Classes ###
## Hyperbolic Tangent Layer ##
class Tanh(BasicLayer):
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
class ReLU(BasicLayer):
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
class SoftPlus(BasicLayer):
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
class Sigmoid(BasicLayer):
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

## Softmax Layer ##        
class Softmax(Layer):
    """Layer with softmax activation.
    
    Provides the following method:
    function -- softmax activation function
    """
    def __init__(self, rng: "presents uniform interface for layers",
                       inputs: "integer",
                       outputs: "integer"):
        """Set activation function and weights and use parent's __init__ method 
           to complete initialization process.
           
        Usage:
        __init__(rng, inputs, outputs)
        
        Arguments:
        rng -- A numpy RandomState.
        inputs -- The number of inputs to this layer.
        outputs -- The number of neurons in this layer.
        """
        # Initialize softmax with zeros for the weights.
        w_values = numpy.zeros((inputs,outputs), dtype = theano.config.floatX)
        self.function = tensor.nnet.softmax
        self.outputs = outputs
        self.inputs = inputs
        super().__init__(w_values)

##### Functions #####
### Reset Layer ID Counter ###        
def reset_layer_ids():
    """Resets the layer id counter to 0."""
    Layer.id = itertools.count()

### Save Network ###
def save_network(net: "Network object", name: "string"):
    """Saves a neural network to the disk.
    
    Usage:
    save_network(net, name)
    
    Arguments:
    net -- A Network object representing a neural net.
    name -- A name to be used for the file.
    """
    filename = name + ".nnet"
    file = open(filename, "wb")
    print("Saving network to " + name + ".nnet")
    pickle.dump(net, file, protocol = pickle.HIGHEST_PROTOCOL)
    file.close()
    print("Done!")
    
### Load Network ###
def load_network(name: "string"):
    """Loads a neural network from the disk.
    
    Usage:
    load_network(name)
    
    Arguments:
    name -- The name of the network file.
    """
    filename = name + ".nnet"
    file = open(filename, "rb")
    print("Loading network " + name + ".nnet")
    net = pickle.Unpickler(file).load()
    file.close()
    print("Done!")
    return net

### Quadratic Cost ###
def quadratic_cost(x,y):
    """Provides the quadratic cost function.
    
    Usage:
    quadratic_cost(x, y)
    
    Arguments:
    x -- The calculated output, a Theano symbol.
    y -- The actual output, a Theano symbol.
    """
    return tensor.sum((x - y) ** 2)
    
### Evaluate Classifiers ###
def eval(net: "Network or ensemble", test_set: "list of lists"):
    """Calculats the classification accuracy of a classifier.
    
    Usage:
    eval(net, test_set)
    
    Arguments:
    net -- A classifier network or ensemble.
    test_set -- The test data set, a list of lists of the form 
                [[data], [intended output]].
    """
    correct = 0
    for item in test_set:
        if numpy.argmax([item[1]]) == net.make_prediction(item[0]):
            correct += 1
    return correct / len(test_set)
    
### Evaluate Autoencoders ###
def eval_autoenc(ae,data):
    sum = 0
    for i in range(0,len(data)):
        sample = data[i][0]
        errs = [abs(i) + abs(j) for i,j in zip(sample,ae.forwardprop([sample]).tolist()[0])]
        sum += numpy.sum(errs)
    return sum / len(data)
