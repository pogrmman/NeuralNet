## Neural Network

### Introduction
This is a module that provides objects and functions that are useful for
constructing neural networks. The simplest way to use it is to go type 
`import neuralnetwork` in an interactive Python shell, and go from there.

### Features
**Supported layer types:** tanh, sigmoid, relu, softplus, softmax

**Supported cost functions:** categorical crossentropy, binary crossentropy, 
quadratic

**Training:** minibatch SGD with optional momentum, optional early stopping

**Regularization:** optional L2

**Evaluation methods:** classification accuracy, autoencoder accuracy, cost
value

**Supported layer initialization methods:** Glorot (for tanh and sigmoid), He
(for relu and softplus)

**Other features:** saving and loading networks to the disk, constructing
networks from arbitrary layers -- even ones trained in other networks,
ensemble networks

### Planned Features
**Layer types:** convolutional, lstm

**Cost functions:** negative log-likelihood

**Regularization:** optional dropout, optional minibatch normalization

**Evaluation Methods:** better autoencoder accuracy

**Layer initialization methods:** layer sequential unit variance 
(Mishkin, Matas 2016)


### Usage
First install [Theano ~~0.9.0~~ 0.8.2](http://deeplearning.net/software/theano/) and 
[Numpy](http://www.numpy.org/). Theano 0.9.0 works fine, but has a memory leak issue.
It's not noticable on small networks, but on large ones, it can lead to crashing the
system. You can configure Theano however you'd like, however, this code has not been
tested on a GPU, so procede with caution. 

You can then clone the repository with 
`git clone https://github.com/pogrmman/NeuralNet.git`

Run an interactive Python shell in the directory, and type 
`import neuralnetwork`

This repository contains two datasets frequently used for machine learning 
tasks - the [iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) and the 
[abalone dataset](https://archive.ics.uci.edu/ml/datasets/abalone). The 
directories for each dataset contain pickles of the dataset, divided into
a test set and a training set (and a validation set for the abalone dataset). 
The pickles are stored in a form usable by Network objects provided by 
neuralnetwork.py. You can access the datasets as follows:
```
f = open("$FILENAME$.pkl", "rb")
$VARIABLE$ = pickle.Unpickler(f).load()
f.close()
```

To create a basic neural network, you use a command like 
`net = neuralnetwork.Network(net_description,learning_rate)` where
net_description is a list of tuples of the form `(number_of_nodes, layer_type)`
that describe the network.

You can train a network with `Network.train($TRAINING_DATA$,$EPOCHS$)`

All functions, classes, and methods have a docstring containing usage
information, so `help()` can be used to find usage information.

### Dependencies
This is built with [Theano ~~0.9.0~~ 0.8.2](http://deeplearning.net/software/theano/),
[Numpy 1.9.2](http://www.numpy.org/), 
and [Python 3.4.3/3.5.3](https://www.python.org/).
