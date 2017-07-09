#!/bin/python3

##### Importing Modules #####
### Builtin Modules ###
import random
### 3rd Party Modules ###
import numpy
import scipy.ndimage
### Package Modules ###
import neuralnet

##### Classes #####
### Augmented Network Class ###
class AugmentedNetwork(neuralnet.Network):
    """Class for neural networks with automatic augmentation for 2d inputs.

    Provives all of the public methods and attributes of the Network class.
    
    Additionally, it provides the following public attributes:
    x_dim -- The input width.
    y_dim -- The input height.
    rotate -- The maximum degree of rotation.
    shift -- The maximum degree of translation.
    shear -- The maximum amount of shearing.
    """
    def __init__(self, x_dim, y_dim, rotate_amount, shift_amount, shear_amount, *args, **kwargs):
        """Initialization for a data-augmented network.
        
        Usage:
        __init__(x_dim, y_dim, rotate_amount, shift_amount, shear_amount, *args, **kwargs)

        Arguments:
        x_dim -- The width of the input.
        y_dim -- The height of the input.
        rotate_amount -- The maximum angle of rotation.
        shift_amount -- The maximum translation.
        shear_amount -- The maximum shear.
        *args -- Positional arguments for instantiation of the Network class.
        **kwargs -- Keyword argumeents for instantiation of the Network class.
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.rotate = rotate_amount
        self.shift = shift_amount
        self.shear = shear_amount
        super().__init__(*args, **kwargs)

    def _make_minibatch(self, data, minbatch_size):
        batch = [random.choice(data).copy() for i in range(0, minbatch_size)]
        for i, item in enumerate(batch):
            batch[i][0] = [item[0][j:j+self.x_dim] for j in range(0,self.x_dim*self.y_dim,self.x_dim)]
            batch[i][0] = numpy.asarray(batch[i][0])
        for i, item in enumerate(batch):
            x_shft = random.randint(-self.shift, self.shift)
            y_shft = random.randint(-self.shift, self.shift)
            rotation = numpy.random.uniform(-self.rotate, self.rotate)
            x_shear = numpy.random.uniform(-self.shear, self.shear)
            y_shear = numpy.random.uniform(-self.shear, self.shear)
            batch[i][0] = transform_array(item[0], x_shft, y_shft, rotation, x_shear, y_shear)
            batch[i][0] = batch[i][0].flatten()
        inpts = [item[0] for item in batch]
        otpts = [item[1] for item in batch]
        return (inpts,otpts)

##### Functions #####
### Array Transformation ###
def transform_array(array, x_shft, y_shft, rotation, x_shear, y_shear):
    """Function for applying a transformation to an array.
    
    Usage:
    transform_array(array, x_shft, y_shft, rotation, x_shear, y_shear)
    
    Arguments:
    array -- The array to be transformed.
    x_shft -- The horizontal translation.
    y_shft -- The vertical translation.
    rotation -- The rotation.
    x_shear -- The horizontal shearing factor.
    y_shear -- The vertical shearing factor.
    """
    shear_matrix = numpy.asarray([[1, x_shear],
                                  [y_shear, 1]])
    array = scipy.ndimage.rotate(array, rotation, reshape = False)
    return scipy.ndimage.interpolation.affine_transform(array, shear_matrix, offset = (x_shft, y_shft))
