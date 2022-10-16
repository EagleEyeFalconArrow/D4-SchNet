import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.ops.array_grad import _TileGrad
from tensorflow.python.framework import ops


def shape(x):
    """
    Get shape of tensor.
    
    :param x: tensor
    :return: shape
    """
    if isinstance(x, tf.Tensor):
        return x.get_shape().as_list()
    return np.shape(x)


@ops.RegisterGradient("TileDense")
def tile_grad_dense(op, grad):
    # This was never really used in the code, so I'm not sure if it's correct, and why it is here.
    """
    Tile gradient for dense tensors.
    
    :param op: tile operation
    :param grad: gradient
    :return: gradient
    """   
    grad = tf.convert_to_tensor(grad)
    return _TileGrad(op, grad)
