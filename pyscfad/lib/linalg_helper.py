import scipy.linalg
from jax import custom_jvp
from pyscfad.lib import numpy as np
from pyscfad.lib import ops

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

@custom_jvp
def _eigh(a, b, lower=True,
          turbo=True, check_finite=True, driver=None):
    if scipy.__version__ >= "1.5.0":
        w, v = scipy.linalg.eigh(a, b=b, lower=lower, check_finite=check_finite, driver=driver)
    else:
        w, v = scipy.linalg.eigh(a, b=b, lower=lower, check_finite=check_finite, turbo=turbo)
    return w, v

@_eigh.defjvp
def _eigh_jvp(primals, tangents):
    deg_thresh = DEG_THRESH
    a, b = primals[:2]
    at, bt = tangents[:2]

    w, v = _eigh(*primals)

    vt_at_v = np.dot(v.conj().T, np.dot(at, v))
    if b is None:
        vt_bt_v = 0
    else:
        vt_bt_v = np.dot(v.conj().T, np.dot(bt, v))
    vt_bt_v_w = np.dot(vt_bt_v, np.diag(w))
    da_minus_ds = vt_at_v - vt_bt_v_w
    dw = np.diag(da_minus_ds)

    eye_n = np.eye(a.shape[-1], dtype=a.dtype)
    eji = w[..., np.newaxis, :] - w[..., np.newaxis]
    idx = abs(eji) < deg_thresh
    eji = ops.index_update(eji, ops.index[idx], 1.e200)
    eji = ops.index_update(eji, np.diag_indices_from(eji), 1.)
    Fmat = np.reciprocal(eji) - eye_n
    dv = np.dot(v, np.multiply(Fmat, da_minus_ds) - np.multiply(eye_n, vt_bt_v) * .5)
    return (w,v), (dw,dv)
