from keras_core import ops

stop_gradient = ops.stop_gradient

def convert_to_numpy(x):
    x = stop_gradient(x)
    return ops.convert_to_numpy(x)

convert_to_tensor = ops.convert_to_tensor
