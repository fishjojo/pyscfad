import numpy as np
import jax

def is_tensor(x):
    return isinstance(x, jax.Array)

def stop_gradient(x):
    return jax.lax.stop_gradient(x)

def convert_to_numpy(x, copy=False):
    x = stop_gradient(x)
    if copy:
        return np.array(x)
    else:
        return np.asarray(x)

def convert_to_tensor(x, dtype=None, **kwargs):
    return jnp.asarray(x, dtype=dtype, **kwargs)
