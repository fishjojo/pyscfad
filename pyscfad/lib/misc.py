import numpy
import jax
from pyscfad import numpy as np
from .jax_helper import jit

def isarray(a):
    return isinstance(a, (numpy.ndarray, jax.Array))

@jit
def square_mat_in_trilu_indices(n):
    tril2sq = np.zeros((n,n), dtype=int)
    idx = np.tril_indices(n)
    idx_flat = np.arange(n*(n+1)//2)
    tril2sq = tril2sq.at[idx[1],idx[0]].set(idx_flat)
    tril2sq = tril2sq.at[idx[0],idx[1]].set(idx_flat)
    return tril2sq
