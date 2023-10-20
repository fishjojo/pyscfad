from keras_core import ops

is_tensor = ops.is_tensor
stop_gradient = ops.stop_gradient
convert_to_tensor = ops.convert_to_tensor

def convert_to_numpy(x):
    x = stop_gradient(x)
    return ops.convert_to_numpy(x)
