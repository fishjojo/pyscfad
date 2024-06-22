from functools import partial
import numpy
import scipy
import scipy.linalg
from jax import numpy as np
from jax import scipy as jax_scipy
from pyscfad.ops import custom_jvp, jit

# default threshold for degenerate eigenvalues
DEG_THRESH = 1e-9

# pylint: disable = redefined-builtin
def eigh(a, b=None, lower=True, eigvals_only=False, overwrite_a=False,
         overwrite_b=False, turbo=True, eigvals=None, type=1,
         check_finite=True, subset_by_index=None, subset_by_value=None,
         driver=None, deg_thresh=DEG_THRESH):
    if overwrite_a is True or overwrite_b is True:
        raise NotImplementedError('Overwritting a or b is not implemeneted.')
    if type != 1:
        raise NotImplementedError('Only the type=1 case of eigh is implemented.')
    if not(eigvals is None and subset_by_index is None and subset_by_value is None):
        raise NotImplementedError('Subset of eigen values is not implemented.')

    a = 0.5 * (a + a.T.conj())
    if b is not None:
        b = 0.5 * (b + b.T.conj())

    w, v = _eigh(a, b, deg_thresh=deg_thresh)

    if eigvals_only:
        return w
    else:
        return w, v

@partial(custom_jvp, nondiff_argnums=(2,))
def _eigh(a, b, deg_thresh=DEG_THRESH):
    w, v = scipy.linalg.eigh(a, b=b)
    w = np.asarray(w, dtype=v.dtype)
    return w, v

@_eigh.defjvp
def _eigh_jvp(deg_thresh, primals, tangents):
    a, b = primals
    at, bt = tangents
    w, v = _eigh(a, b, deg_thresh)

    eji = w[None, :] - w[:, None]
    idx = numpy.asarray(abs(eji) <= deg_thresh, dtype=bool)
    eji = eji.at[idx].set(1e200)
    eji = eji.at[numpy.diag_indices_from(eji)].set(1)
    Fmat = 1 / eji - numpy.eye(a.shape[-1])
    if b is None:
        dw, dv = _eigh_jvp_jitted_nob(v, Fmat, at)
    else:
        bmask = numpy.zeros(a.shape)
        bmask[idx] = 1
        dw, dv = _eigh_jvp_jitted(w, v, Fmat, at, bt, bmask)
    return (w, v), (dw, dv)

@jit
def _eigh_jvp_jitted(w, v, Fmat, at, bt, bmask):
    vt_at_v = np.dot(v.conj().T, np.dot(at, v))
    vt_bt_v = np.dot(v.conj().T, np.dot(bt, v))
    vt_bt_v_w = np.dot(vt_bt_v, np.diag(w))
    da_minus_ds = vt_at_v - vt_bt_v_w
    dw = np.diag(da_minus_ds)#.real

    dv = np.dot(v, np.multiply(Fmat, da_minus_ds) - np.multiply(bmask, vt_bt_v) * .5)
    return dw, dv

@jit
def _eigh_jvp_jitted_nob(v, Fmat, at):
    vt_at_v = np.dot(v.conj().T, np.dot(at, v))
    dw = np.diag(vt_at_v)
    dv = np.dot(v, np.multiply(Fmat, vt_at_v))
    return dw, dv


def svd(a, full_matrices=True, compute_uv=True,
        overwrite_a=False, check_finite=True,
        lapack_driver='gesdd'):
    if not full_matrices or not compute_uv:
        return jax_scipy.linalg.svd(a,
                                    full_matrices=full_matrices,
                                    compute_uv=compute_uv)
    else:
        return _svd(a)

@custom_jvp
def _svd(a):
    return jax_scipy.linalg.svd(a)

@_svd.defjvp
def _svd_jvp(primals, tangents):
    A, = primals
    dA, = tangents
    if np.iscomplexobj(A):
        raise NotImplementedError

    m, n = A.shape
    if m > n:
        raise NotImplementedError('Use svd(A.conj().T) instead.')

    U, s, Vt = _svd(A)
    Ut = U.conj().T
    V = Vt.conj().T
    s_dim = s[None, :]

    dS = Ut @ dA @ V
    ds = np.diagonal(dS, 0, -2, -1).real

    s_diffs = (s_dim + s_dim.T) * (s_dim - s_dim.T)
    s_diffs_zeros = (s_diffs == 0).astype(s_diffs.dtype)
    F = 1. / (s_diffs + s_diffs_zeros) - s_diffs_zeros

    dP1 = dS[:,:m]
    dP2 = dS[:,m:]
    dSS = dP1 * s_dim
    SdS = s_dim.T * dP1

    dU = U @ (F * (dSS + dSS.conj().T))
    dD1 = F * (SdS + SdS.conj().T)

    s_zeros = (s == 0).astype(s.dtype)
    s_inv = 1. / (s + s_zeros) - s_zeros
    dD2 = s_inv[:,None] * dP2

    dV = np.zeros_like(V)
    dV = dV.at[:m,:m].set(dD1)
    dV = dV.at[:m,m:].set(-dD2)
    dV = dV.at[m:,:m].set(dD2.conj().T)
    dV = V @ dV
    return (U, s, Vt), (dU, ds, dV.conj().T)
