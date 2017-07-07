#!/bin/python3

##### Importing Modules #####
### 3rd Party Modules ###
import numpy
### Package Modules ###
import neuralnet

class DataAugmentedNetwork(neuralnet.Network):
    def _data_generator(self):
        # Overwrite here and add augmentation
