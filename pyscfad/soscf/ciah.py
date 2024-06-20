import numpy
# TODO add other backend for expm
from jax.scipy.linalg import expm
from pyscfad import numpy as np
from pyscfad.ops import jit

def pack_uniq_var(mat):
    nmo = mat.shape[0]
    idx = np.tril_indices(nmo, -1)
    return mat[idx]

@jit
def unpack_uniq_var(v):
    nmo = int(numpy.sqrt(v.size*2)) + 1
    idx = np.tril_indices(nmo, -1)
    mat = np.zeros((nmo,nmo))
    mat = mat.at[idx].set(v)
    return mat - mat.conj().T

def extract_rotation(dr, u0=1):
    dr = unpack_uniq_var(dr)
    return np.dot(u0, expm(dr))
