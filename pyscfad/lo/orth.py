from functools import partial, reduce
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad import scipy

@partial(ops.jit, static_argnums=1)
def lowdin(s, thresh=1e-15):
    e, v = scipy.linalg.eigh(s)
    e_sqrt = np.where(e>thresh, np.sqrt(e), np.inf)
    return np.dot(v/e_sqrt[None,:], v.conj().T)

def vec_lowdin(c, s=1):
    return np.dot(c, lowdin(reduce(np.dot, (c.conj().T, s, c))))

