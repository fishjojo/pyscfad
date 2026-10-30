# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import math
import jax
from pyscf.lib.numpy_helper import (
    PLAIN,
    HERMITIAN,
    ANTIHERMI,
    SYMMETRIC,
)
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.ops import jit, vmap

__all__ = [
    'PLAIN',
    'HERMITIAN',
    'ANTIHERMI',
    'SYMMETRIC',
    'unpack_triu',
    'unpack_tril',
    'pack_tril',
    'hermi_triu',
    'cartesian_prod',
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

# The packing indices are computed in the traced program, so that no index
# array becomes a constant of the compiled functions or of their jitted callers.
def _tril_position(n):
    '''``(n, n)`` packed positions of ``(max(i,j), min(i,j))``, in PySCF
    order ``t = i*(i+1)/2 + j`` for ``j <= i``.'''
    i = np.arange(n, dtype=np.int64)
    hi = np.maximum(i[:,None], i)
    lo = np.minimum(i[:,None], i)
    return hi * (hi + 1) // 2 + lo

def _tril_flat_index(n):
    '''Flattened positions ``i*n + j`` of the packed entries ``t``.'''
    t = np.arange(n * (n + 1) // 2, dtype=np.int64)
    # row of t, i = floor((sqrt(8t+1) - 1)/2); exact in float64 (pyscfad
    # always enables x64) for rows up to at least 1e8, also at the row
    # boundaries
    i = np.floor((np.sqrt(8. * t + 1.) - 1.) / 2.).astype(np.int64)
    # i*n + j with j = t - i*(i+1)/2
    return t + i * (2 * n - i - 1) // 2

@partial(jit, static_argnums=1)
def _unpack_tril_gather(tril, filltriu):
    n = math.isqrt(2 * tril.shape[-1])
    # the lower triangle, diagonal included, and its mirror
    out = np.take(tril, _tril_position(n), axis=-1, mode='clip')
    if filltriu == SYMMETRIC or (filltriu == HERMITIAN and not np.iscomplexobj(tril)):
        return out
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    if filltriu == HERMITIAN:
        return np.where(upper, out.conj(), out)
    if filltriu == ANTIHERMI:
        return np.where(upper, -out.conj(), out)
    if filltriu == PLAIN:
        return np.where(upper, 0, out)
    raise KeyError(f'filltriu={filltriu}')

@partial(jit, static_argnums=1)
def _unpack_tril_scatter(tril, filltriu):
    '''The map of :func:`_unpack_tril_gather`, scattering the lower triangles.'''
    n = math.isqrt(2 * tril.shape[-1])
    low = np.zeros(tril.shape[:-1] + (n * n,), dtype=tril.dtype)
    low = low.at[..., _tril_flat_index(n)].set(tril, unique_indices=True,
                                                 indices_are_sorted=True)
    low = low.reshape(tril.shape[:-1] + (n, n))
    if filltriu == PLAIN:
        return low
    up = np.triu(np.swapaxes(low, -1, -2), 1)
    if filltriu == SYMMETRIC:
        return low + up
    if filltriu == HERMITIAN:
        return low + up.conj()
    if filltriu == ANTIHERMI:
        return low - up.conj()
    raise KeyError(f'filltriu={filltriu}')

@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _unpack_tril(tril, filltriu):
    return _unpack_tril_gather(tril, filltriu)

@_unpack_tril.defjvp
def _unpack_tril_jvp(filltriu, primals, tangents):
    tril, = primals
    tril_dot, = tangents
    # the tangent in the scatter form: reverse mode transposes it into a
    # gather, whereas the transpose of the gather is a scatter-add
    return (_unpack_tril_gather(tril, filltriu),
            _unpack_tril_scatter(tril_dot, filltriu))

def unpack_tril(tril, filltriu=HERMITIAN, axis=-1, out=None):
    '''Unpack the lower triangles stored along ``axis``.

    See also:
        :func:`pyscf.lib.numpy_helper.unpack_tril`

    Notes:
        ``out`` is unused.
    '''
    del out
    axis = axis % tril.ndim
    npair = tril.shape[axis]
    n = math.isqrt(2 * npair)
    if n * (n + 1) // 2 != npair:
        raise ValueError(f'{npair} is not the size of a lower triangle')
    return _unpack_tril_along(tril, filltriu, axis)

@partial(jit, static_argnums=(1, 2))
def _unpack_tril_along(tril, filltriu, axis):
    last = axis == tril.ndim - 1
    if not last:
        tril = np.moveaxis(tril, axis, -1)
    mat = _unpack_tril(tril, filltriu)
    if not last:
        mat = np.moveaxis(mat, (-2, -1), (axis, axis + 1))
    return mat

def pack_tril(a, axis=-1, out=None):
    '''Flatten the lower triangles in either the last
    (``axis=-1``) or the first (``axis=0``) two axes of ``a``.

    See also:
        :func:`pyscf.lib.numpy_helper.pack_tril`

    Notes:
        ``out`` is unused.
    '''
    del out
    if axis == 0:
        n = a.shape[0]
        if a.shape[1] != n:
            raise ValueError(f'leading two axes of shape {a.shape} are not square')
    elif axis == -1:
        n = a.shape[-1]
        if a.shape[-2] != n:
            raise ValueError(f'last two axes of shape {a.shape} are not square')
    else:
        raise KeyError(f'axis={axis} not supported')
    return _pack_tril(a, axis)

@partial(jit, static_argnums=1)
def _pack_tril(a, axis):
    if axis == 0:
        n = a.shape[0]
        return np.take(a.reshape((n * n,) + a.shape[2:]), _tril_flat_index(n),
                       axis=0, mode='clip')
    n = a.shape[-1]
    return np.take(a.reshape(a.shape[:-2] + (n * n,)), _tril_flat_index(n),
                   axis=-1, mode='clip')

@partial(jit, static_argnums=1)
def hermi_triu(a, hermi=HERMITIAN):
    '''Batched :func:`~pyscf.lib.numpy_helper.hermi_triu`.
    '''
    assert hermi in (HERMITIAN, ANTIHERMI), f'hermi={hermi} not supported'
    shape = a.shape
    n, m = shape[-2], shape[-1]
    assert n == m, f'n ({n}) != m ({m})'

    triu_idx = np.triu_indices(n, 1)
    def fn(mat):
        if hermi == HERMITIAN:
            triu = np.tril(mat, -1).T.conj()
        elif hermi == ANTIHERMI:
            triu = -np.tril(mat, -1).T.conj()
        return ops.index_update(mat, triu_idx, triu[triu_idx])

    out = vmap(fn, signature='(n,n)->(n,n)')(a.reshape(-1,n,n))
    return out.reshape(shape)

@jit
def cartesian_prod(arrays):
    '''Cartesian product of a list of 1D arrays
    '''
    grids = np.meshgrid(*arrays, indexing='ij')
    stacked = np.stack(grids, axis=-1)
    return stacked.reshape(-1, len(arrays))
