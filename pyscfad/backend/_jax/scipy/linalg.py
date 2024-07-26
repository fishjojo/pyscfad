"""
Custom jax.scipy.linalg functions
"""
import warnings
import jax
from jax import numpy as jnp
from jax import scipy as jsp
from jax._src.numpy.util import promote_dtypes_inexact
from jax._src.lax.linalg import _T, _H
from ..lax import linalg as lax_linalg

def eigh(a, b=None, *,
         lower=True,
         eigvals_only=False,
         overwrite_a=False,
         overwrite_b=False,
         type=1,
         check_finite=False,
         subset_by_index=None,
         subset_by_value=None,
         driver=None,
         deg_thresh=1e-9):
    if overwrite_a or overwrite_b:
        warnings.warn('Arguments \'overwrite_a\' and \'overwrite_b\' have no effect.')
    if check_finite:
        warnings.warn('Argument \'check_finite\' has no effect.')
    if subset_by_index or subset_by_value:
        raise NotImplementedError('Computing subset of eigenvalues is not supported.')
    if driver:
        warnings.warn('Argument \'driver\' has no effect.')

    del (overwrite_a, overwrite_b, check_finite,
         subset_by_index, subset_by_value, driver)
    return _eigh(a, b, lower, type, eigvals_only, deg_thresh)

def _eigh(a, b, lower, itype, eigvals_only, deg_thresh):
    if b is None:
        b = jnp.zeros_like(a) + jnp.eye(a.shape[-1])

    a, b = promote_dtypes_inexact(jnp.asarray(a), jnp.asarray(b))
    w, v = lax_linalg.eigh_gen(a, b, lower=lower, itype=itype, deg_thresh=deg_thresh)

    if eigvals_only:
        return w
    else:
        return w, v

def svd(a, full_matrices=True, compute_uv=True,
        overwrite_a=False, check_finite=False,
        lapack_driver=None):
    if overwrite_a:
        warnings.warn('Argument \'overwrite_a\' has no effect.')
    if check_finite:
        warnings.warn('Argument \'check_finite\' has no effect.')
    if lapack_driver:
        warnings.warn('Argument \'lapack_driver\' has no effect.')
    del overwrite_a, check_finite, lapack_driver

    if not full_matrices or not compute_uv:
        return jsp.linalg.svd(a,
                              full_matrices=full_matrices,
                              compute_uv=compute_uv)
    else:
        return _svd(a)

@jax.custom_jvp
def _svd(a):
    return jsp.linalg.svd(a, full_matrices=True, compute_uv=True)

@_svd.defjvp
def _svd_jvp(primals, tangents):
    A, = primals
    dA, = tangents
    m, n = A.shape
    if m > n:
        raise NotImplementedError('Use svd(A.conj().T) instead.')

    U, s, Vt = _svd(A)
    Ut = _H(U)
    V = _H(Vt)
    s_dim = s[..., jnp.newaxis, :]

    dS = Ut @ dA @ V
    ds = jnp.diagonal(dS, 0, -2, -1).real

    s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
    s_diffs_zeros = (s_diffs == 0).astype(s.dtype)
    F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros

    dP1 = dS[..., :, :m]
    dP2 = dS[..., :, m:]
    dSS = dP1 * s_dim
    SdS = _T(s_dim) * dP1

    dU = U @ (F * (dSS + _H(dSS)))
    dD1 = F * (SdS + _H(SdS))

    s_zeros = (s == 0).astype(s.dtype)
    s_inv = 1 / (s + s_zeros) - s_zeros
    dD2 = s_inv[..., :, jnp.newaxis] * dP2

    dV = jnp.zeros_like(V)
    dV = dV.at[..., :m, :m].set(dD1)
    dV = dV.at[..., :m, m:].set(-dD2)
    dV = dV.at[..., m:, :m].set(dD2.conj().T)
    dV = V @ dV
    return (U, s, Vt), (dU, ds, _H(dV))

