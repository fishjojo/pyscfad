from functools import partial
from jax import jit, custom_jvp
from pyscf.dft import libxc
from pyscf.dft.libxc import parse_xc, is_lda, is_meta_gga
from pyscfad.lib import numpy as np

def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    # NOTE only consider exc and vxc
    if deriv > 1:
        raise NotImplementedError

    hyb, fn_facs = parse_xc(xc_code)
    if omega is not None:
        hyb[2] = float(omega)

    exc = _eval_xc_comp(rho, hyb, fn_facs, spin, relativity, deriv=0, verbose=verbose)
    if deriv == 0:
        vxc = (None,) * 4
    else:
        vxc = _eval_xc_comp(rho, hyb, fn_facs, spin, relativity, deriv=1, verbose=verbose)
    return exc, vxc, None, None

@partial(custom_jvp, nondiff_argnums=tuple(range(1,7)))
def _eval_xc_comp(rho, hyb, fn_facs, spin=0, relativity=0, deriv=1, verbose=None):
    return libxc._eval_xc(hyb, fn_facs, rho, spin, relativity, deriv, verbose)[deriv]

@_eval_xc_comp.defjvp
def _eval_xc_comp_jvp(hyb, fn_facs, spin, relativity, deriv, verbose,
                      primals, tangents):
    rho, = primals
    rho_t, = tangents
    if deriv > 2:
        raise NotImplementedError

    val  = _eval_xc_comp(rho, hyb, fn_facs, spin, relativity, deriv, verbose)
    val1 = _eval_xc_comp(rho, hyb, fn_facs, spin, relativity, deriv+1, verbose)

    fn_ids = [x[0] for x in fn_facs]
    n = len(fn_ids)
    if (n == 0 or
        all((is_lda(x) for x in fn_ids))):
        if deriv == 0:
            jvp = (val1[0] - val) / rho * rho_t
        elif deriv == 1:
            jvp = (val1[0] * rho_t,) + (None,) * 3
        else:
            jvp = (val1[0] * rho_t,) + (None,) * 9
    elif any((is_meta_gga(x) for x in fn_ids)):
        if deriv == 0:
            exc1 = _exc_partial_deriv(rho, val, val1, "MGGA")
            jvp = np.einsum('np,np->p', exc1, rho_t)
        elif deriv == 1:
            vrho1, vsigma1, vlapl1, vtau1 = _vxc_partial_deriv(rho, val, val1, "MGGA")
            vrho_jvp = np.einsum('np,np->p', vrho1, rho_t)
            vsigma_jvp = np.einsum('np,np->p', vsigma1, rho_t)
            vlapl_jvp = np.einsum('np,np->p', vlapl1, rho_t)
            vtau_jvp = np.einsum('np,np->p', vtau1, rho_t)
            vrho1 = vsigma1 = vlapl1 = vtau1 = None
            jvp = np.vstack((vrho_jvp, vsigma_jvp, vlapl_jvp, vtau_jvp))
        else:
            raise NotImplementedError
    else:
        if deriv == 0:
            exc1 = _exc_partial_deriv(rho, val, val1, "GGA")
            jvp = np.einsum('np,np->p', exc1, rho_t)
        elif deriv == 1:
            vrho1, vsigma1 = _vxc_partial_deriv(rho, val, val1, "GGA")[:2]
            vrho_jvp = np.einsum('np,np->p', vrho1, rho_t)
            vsigma_jvp = np.einsum('np,np->p', vsigma1, rho_t)
            vrho1 = vsigma1 = None
            jvp = (vrho_jvp, vsigma_jvp, None, None)
        else:
            v2rho2, v2rhosigma, v2sigma2 = _fxc_partial_deriv(rho, val, val1, "GGA")[:3]
            v2rho2_jvp = np.einsum('np,np->p', v2rho2, rho_t)
            v2rhosigma_jvp = np.einsum('np,np->p', v2rhosigma, rho_t)
            v2sigma2_jvp = np.einsum('np,np->p', v2sigma2, rho_t)
            jvp = (v2rho2_jvp, v2rhosigma_jvp, v2sigma2_jvp) + (None,)*7
    return val, jvp

@partial(jit, static_argnames=['xctype'])
def _exc_partial_deriv(rho, exc, vxc, xctype="LDA"):
    if xctype == "LDA":
        exc1 = (vxc[0] - exc) / rho
    elif xctype in ["GGA", "MGGA"]:
        drho = (vxc[0] - exc) / rho[0]
        dsigma = vxc[1] / rho[0] * 2. * rho[1:4]
        exc1 = np.vstack((drho, dsigma))
        if xctype == "MGGA":
            dlap = vxc[2] / rho[0]
            dtau = vxc[3] / rho[0]
            exc1 = np.vstack((exc1, dlap, dtau))
    else:
        raise KeyError
    return exc1

@partial(jit, static_argnames=['xctype'])
def _vxc_partial_deriv(rho, vxc, fxc, xctype="LDA"):
    vrho1 = vsigma1 = vlapl1 = vtau1 = None
    if xctype == "LDA":
        vrho1 = fxc[0]
    elif xctype in ["GGA", "MGGA"]:
        vrho1 = np.vstack((fxc[0], fxc[1] * 2. * rho[1:4]))
        vsigma1 = np.vstack((fxc[1], fxc[2] * 2. * rho[1:4]))
        if xctype == "MGGA":
            vrho1 = np.vstack((vrho1, fxc[5], fxc[6]))
            vsigma1 = np.vstack((vsigma1, fxc[8], fxc[9]))
            vlapl1 = np.vstack((fxc[5], fxc[8] * 2. * rho[1:4], fxc[3], fxc[7]))
            vtau1 = np.vstack((fxc[6], fxc[9] * 2. * rho[1:4], fxc[7], fxc[4]))
    else:
        raise KeyError
    return vrho1, vsigma1, vlapl1, vtau1

@partial(jit, static_argnames=['xctype'])
def _fxc_partial_deriv(rho, fxc, kxc, xctype="LDA"):
    v2rho2_1 = v2rhosigma_1 = v2sigma2_1 = None
    if xctype == "LDA":
        v2rho2_1 = kxc[0]
    elif xctype in ["GGA", "MGGA"]:
        v2rho2_1 = np.vstack((kxc[0], kxc[1] * 2. * rho[1:4]))
        v2rhosigma_1 = np.vstack((kxc[1], kxc[2] * 2. * rho[1:4]))
        v2sigma2_1 = np.vstack((kxc[2], kxc[3] * 2. * rho[1:4]))
        if xctype == "MGGA":
            raise NotImplementedError
    else:
        raise KeyError
    return (v2rho2_1, v2rhosigma_1, v2sigma2_1) + (None,) * 7
