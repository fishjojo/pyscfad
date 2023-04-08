import numpy
import jax

def isarray(a):
    return isinstance(a, (numpy.ndarray, jax.Array))
