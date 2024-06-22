from functools import partial, reduce
import numpy
import scipy
from jax import scipy as jax_scipy
from pyscfad import numpy as np
from pyscfad.ops import custom_jvp

@partial(custom_jvp, nondiff_argnums=(1,))
def lowdin(s, thresh=1e-15):
    e, v = scipy.linalg.eigh(s)
    idx = e > thresh
    return numpy.dot(v[:,idx]/numpy.sqrt(e[idx]), v[:,idx].conj().T)

@lowdin.defjvp
def lowdin_jvp(thresh, primals, tangents):
    s, = primals
    ds, = tangents

    e, u = jax_scipy.linalg.eigh(s)
    idx = e > thresh
    e = e[idx]
    u = u[:,idx]
    sqrt_e = np.sqrt(e)
    primal_out = np.dot(u/sqrt_e, u.conj().T)

    ut_ds_u = reduce(np.dot, (u.conj().T, ds, u))
    denom = e[:,None] * sqrt_e[None,:]
    ut_ds_u = ut_ds_u / (denom + denom.T)
    jvp = -reduce(np.dot, (u, ut_ds_u, u.conj().T))
    return primal_out, jvp

def vec_lowdin(c, s=1):
    return np.dot(c, lowdin(reduce(np.dot, (c.conj().T,s,c))))
