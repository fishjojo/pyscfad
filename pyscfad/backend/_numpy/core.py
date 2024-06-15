import numpy as np

def is_tensor(x):
    return isinstance(x, (np.ndarray, np.generic))

def stop_gradient(x):
    return x

def convert_to_numpy(x, copy=False):
    if copy:
        return np.array(x)
    else:
        return np.asarray(x)

def convert_to_tensor(x, dtype=None, **kwargs):
    return np.asarray(x, dtype=dtype, **kwargs)
