from functools import partial
import numpy
import jax
from pyscfad.lib import numpy as jnp
from pyscfad.lib import ops

@partial(jax.jit, static_argnums=1)
def unpack_triu(triu, filltril=0):
    '''
    Unpack the upper triangular part of a matrix
    '''
    assert triu.ndim == 1
    nd = int(numpy.sqrt(2*triu.size))
    shape = (nd, nd)
    out = jnp.empty(shape, dtype=triu.dtype)
    idx = numpy.triu_indices(nd)
    out = ops.index_update(out, idx, triu)
    if filltril == 0:
        return out
    elif filltril == 1:
        out = out + out.conj().T
        out = ops.index_mul(out, numpy.diag_indices(nd), .5)
        return out
    elif filltril == 2:
        out = out - out.conj().T
        return out
    else:
        raise KeyError
