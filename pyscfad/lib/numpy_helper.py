from functools import partial
import numpy as onp
#from jax import numpy
#from jax.config import config as jax_config
from pyscf import numpy
from pyscf.lib import ops
from .jax_helper import jit
#jax_config.update("jax_enable_x64", True)

einsum = numpy.einsum
dot = numpy.dot

__all__ = ['numpy', 'einsum', 'dot',
           'PLAIN', 'HERMITIAN', 'ANTIHERMI', 'SYMMETRIC',
           'unpack_triu', 'unpack_tril',]

PLAIN = 0
HERMITIAN = 1
ANTIHERMI = 2
SYMMETRIC = 3

@partial(jit, static_argnums=1)
def unpack_triu(triu, filltril=PLAIN):
    '''
    Unpack the upper triangular part of a matrix
    '''
    assert triu.ndim == 1
    nd = int(onp.sqrt(2*triu.size))
    out = numpy.zeros((nd,nd), dtype=triu.dtype)
    idx = onp.triu_indices(nd)
    out = ops.index_update(out, idx, triu)
    if filltril == PLAIN:
        return out
    elif filltril == HERMITIAN:
        out += numpy.tril(out.T.conj(), -1)
        return out
    elif filltril == ANTIHERMI:
        out -= out.conj().T
        return out
    elif filltril == SYMMETRIC:
        out += numpy.tril(out.T, -1)
        return out
    else:
        raise KeyError

@partial(jit, static_argnums=1)
def unpack_tril(tril, filltriu=PLAIN):
    '''
    Unpack the lower triangular part of a matrix
    '''
    assert tril.ndim == 1
    nd = int(onp.sqrt(2*tril.size))
    out = numpy.zeros((nd,nd), dtype=tril.dtype)
    idx = onp.tril_indices(nd)
    out = ops.index_update(out, idx, tril)
    if filltriu == PLAIN:
        return out
    elif filltriu == HERMITIAN:
        out += numpy.triu(out.T.conj(), 1)
        return out
    elif filltriu == ANTIHERMI:
        out -= out.T.conj()
        return out
    elif filltriu == SYMMETRIC:
        out += numpy.triu(out.T, 1)
        return out
    else:
        raise KeyError
