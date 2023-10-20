import torch
from keras_core import ops

stop_gradient = ops.stop_gradient
convert_to_tensor = ops.convert_to_tensor

def is_tensor(x):
    return isinstance(x, torch.Tensor)

def convert_to_numpy(x):
    x = stop_gradient(x)
    return ops.convert_to_numpy(x)
