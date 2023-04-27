from functools import reduce
from pyscf import numpy as np
from pyscfad import scipy

def lowdin(s, thresh=1e-12):
    e, v = scipy.linalg.eigh(s)
    idx = e > thresh
    return np.dot(v[:,idx]/np.sqrt(e[idx]), v[:,idx].conj().T)

def vec_lowdin(c, s=1):
    return np.dot(c, lowdin(reduce(np.dot, (c.conj().T,s,c))))
