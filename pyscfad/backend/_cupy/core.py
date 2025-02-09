import cupy

def is_array(x):
    return isinstance(x, cupy.ndarray)

def to_numpy(x):
    return cupy.asnumpy(x)
