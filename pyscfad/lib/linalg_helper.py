from functools import partial
import numpy
import scipy
import scipy.linalg
from pyscf import numpy as np
from pyscfad.lib import ops, custom_jvp, jit

# threshold for degenerate eigenvalues
DEG_THRESH = 1e-10

def eigh(a, b=None, lower=True, eigvals_only=False, overwrite_a=False,
         overwrite_b=False, turbo=True, eigvals=None, type=1,
         check_finite=True, subset_by_index=None, subset_by_value=None,
         driver=None):
    if overwrite_a is True or overwrite_b is True:
        raise NotImplementedError("Overwritting a or b is not implemeneted.")
    if type != 1:
        raise NotImplementedError("Only the type=1 case of eigh is implemented.")
    if not(eigvals is None and subset_by_index is None and subset_by_value is None):
        raise NotImplementedError("Subset of eigen values is not implemented.")

    a = 0.5 * (a + a.T.conj())
    if b is not None:
        b = 0.5 * (b + b.T.conj())

    w, v =  _eigh(a, b, lower=lower,
                  turbo=turbo, check_finite=check_finite, driver=driver)

    if eigvals_only:
        return w
    else:
        return w, v

@partial(custom_jvp, nondiff_argnums=(2,3,4,5))
def _eigh(a, b, lower=True,
          turbo=True, check_finite=True, driver=None):
    if scipy.__version__ >= "1.5.0":
        w, v = scipy.linalg.eigh(a, b=b, lower=lower, check_finite=check_finite, driver=driver)
    else:
        w, v = scipy.linalg.eigh(a, b=b, lower=lower, check_finite=check_finite, turbo=turbo)
    w = np.asarray(w, dtype=v.dtype)
    return w, v

@_eigh.defjvp
def _eigh_jvp(lower, turbo, check_finite, driver, primals, tangents):
    a, b = primals
    at, bt = tangents
    w, v = primal_out = _eigh(*primals,
                              lower=lower, turbo=turbo, check_finite=check_finite, driver=driver)

    deg_thresh = DEG_THRESH
    eji = w[..., numpy.newaxis, :] - w[..., numpy.newaxis]
    idx = abs(eji) < deg_thresh
    #eji[idx] = 1.e200
    #eji[numpy.diag_indices_from(eji)] = 1
    eji = ops.index_update(eji, idx, 1.e200)
    eji = ops.index_update(eji, np.diag_indices_from(eji), 1.)
    eye_n = numpy.eye(a.shape[-1], dtype=a.dtype)
    Fmat = np.reciprocal(eji) - eye_n
    if b is None:
        dw, dv = _eigh_jvp_jitted_nob(v, Fmat, at)
    else:
        dw, dv = _eigh_jvp_jitted(w, v, Fmat, at, bt)
        if idx.sum() > 0:
            mask = np.zeros_like(b, dtype=np.int32)
            mask = mask.at[idx].set(1)
            mask -= eye_n
            tmp = -.5 * np.dot(v.conj().T, np.dot(bt, v))
            dv += np.dot(v, np.multiply(mask, tmp))
    return primal_out, (dw,dv)

@jit
def _eigh_jvp_jitted(w, v, Fmat, at, bt):
    vt_at_v = np.dot(v.conj().T, np.dot(at, v))
    vt_bt_v = np.dot(v.conj().T, np.dot(bt, v))
    vt_bt_v_w = np.dot(vt_bt_v, np.diag(w))
    da_minus_ds = vt_at_v - vt_bt_v_w
    dw = np.diag(da_minus_ds)#.real

    eye_n = np.eye(vt_bt_v.shape[-1])
    dv = np.dot(v, np.multiply(Fmat, da_minus_ds) - np.multiply(eye_n, vt_bt_v) * .5)
    return dw, dv

@jit
def _eigh_jvp_jitted_nob(v, Fmat, at):
    vt_at_v = np.dot(v.conj().T, np.dot(at, v))
    dw = np.diag(vt_at_v)
    dv = np.dot(v, np.multiply(Fmat, vt_at_v))
    return dw, dv
