from functools import partial
import math
from pyscf.lib.numpy_helper import (
    PLAIN,
    HERMITIAN,
    ANTIHERMI,
    SYMMETRIC,
)
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.ops import jit, vmap
from pyscfad import config

__all__ = [
    'PLAIN',
    'HERMITIAN',
    'ANTIHERMI',
    'SYMMETRIC',
    'unpack_triu',
    'unpack_tril',
    'pack_tril'
]

@partial(jit, static_argnums=1)
def _unpack_triu(triu, filltril=HERMITIAN):
    '''
    Unpack the upper triangular part of a matrix
    '''
    assert triu.ndim == 1
    nd = int(math.sqrt(2*triu.size))
    out = np.zeros((nd,nd), dtype=triu.dtype)
    idx = np.triu_indices(nd)
    out = ops.index_update(out, idx, triu)
    if filltril == PLAIN:
        return out
    elif filltril == HERMITIAN:
        out += np.tril(out.T.conj(), -1)
        return out
    elif filltril == ANTIHERMI:
        out -= out.conj().T
        return out
    elif filltril == SYMMETRIC:
        out += np.tril(out.T, -1)
        return out
    else:
        raise KeyError

def unpack_triu(triu, filltril=HERMITIAN, axis=-1, out=None):
    if triu.ndim == 1:
        out = _unpack_triu(triu, filltril)
    elif triu.ndim == 2:
        if axis == -1 or axis == 1:
            out = vmap(_unpack_triu, (0,None), signature='(n)->(m,m)')(triu, filltril)
        elif axis == 0 or axis == -2:
            out = vmap(_unpack_triu, (1,None), signature='(n)->(m,m)')(triu, filltril)
    else:
        raise NotImplementedError
    return out

@partial(jit, static_argnums=1)
def _unpack_tril(tril, filltriu=HERMITIAN):
    '''
    Unpack the lower triangular part of a matrix
    '''
    assert tril.ndim == 1
    nd = int(math.sqrt(2*tril.size))
    out = np.zeros((nd,nd), dtype=tril.dtype)
    idx = np.tril_indices(nd)
    out = ops.index_update(out, idx, tril)
    if filltriu == PLAIN:
        return out
    elif filltriu == HERMITIAN:
        out += np.triu(out.T.conj(), 1)
        return out
    elif filltriu == ANTIHERMI:
        out -= out.T.conj()
        return out
    elif filltriu == SYMMETRIC:
        out += np.triu(out.T, 1)
        return out
    else:
        raise KeyError

def unpack_tril(tril, filltriu=HERMITIAN, axis=-1, out=None):
    if config.moleintor_opt and axis == -1:
        from pyscfad.lib import _numpy_helper_opt
        return _numpy_helper_opt._unpack_tril(tril, filltriu, axis, out)
    if tril.ndim == 1:
        out = _unpack_tril(tril, filltriu)
    elif tril.ndim == 2:
        if axis == -1 or axis == 1:
            out = vmap(_unpack_tril, (0,None), signature='(n)->(m,m)')(tril, filltriu)
        elif axis == 0 or axis == -2:
            out = vmap(_unpack_tril, (1,None), signature='(n)->(m,m)')(tril, filltriu)
    else:
        raise NotImplementedError
    return out

def pack_tril(a, axis=-1, out=None):
    '''
    Lower triangular part of a matrix as a vector
    '''
    if config.moleintor_opt and axis == -1:
        from pyscfad.lib import _numpy_helper_opt
        return _numpy_helper_opt._pack_tril(a, axis, out)
    def fn(mat):
        idx = np.tril_indices(mat.shape[0])
        return mat[idx].ravel()

    if a.ndim == 3:
        if axis == -1:
            tril = vmap(fn, signature='(m,m)->(n)')(a)
        elif axis == 0:
            tril = vmap(fn, -1, signature='(m,m)->(n)')(a)
        else:
            raise KeyError
    else:
        raise NotImplementedError
    return tril
