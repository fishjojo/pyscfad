from os import error
import warnings
import math
import ctypes
from functools import partial
import numpy
from jax import custom_jvp
from pyscf.dft import libxc
from pyscf.dft.libxc import parse_xc, is_lda, is_meta_gga
from pyscfad.lib import numpy as jnp

def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    hyb, fn_facs = parse_xc(xc_code)
    if omega is not None:
        hyb[2] = float(omega)
    return _eval_xc(rho, hyb, fn_facs, spin, relativity, deriv, verbose)

@partial(custom_jvp, nondiff_argnums=tuple(range(1,7)))
def _eval_xc(rho, hyb, fn_facs, spin=0, relativity=0, deriv=1, verbose=None):
    return libxc._eval_xc(hyb, fn_facs, rho, spin, relativity, deriv, verbose)

@_eval_xc.defjvp
def _eval_xc_jvp(hyb, fn_facs, spin, relativity, deriv, verbose,
                 primals, tangents):
    rho,   = primals
    rho_t, = tangents

    if deriv > 2:
        raise NotImplementedError

    exc, vxc, fxc, kxc = _eval_xc(rho, hyb, fn_facs, spin, relativity, deriv+1, verbose)

    fn_ids = [x[0] for x in fn_facs]
    n      = len(fn_ids)

    fn_is_lda      = (n == 0) or all((is_lda(x) for x in fn_ids))
    fn_is_meta_gga = any((is_meta_gga(x) for x in fn_ids))
    fn_is_gga      = (not fn_is_lda) and (not fn_is_meta_gga)

    if fn_is_lda:
        if spin == 0:
            exc1    = _exc_partial_deriv(rho, exc, vxc, spin, xctype="LDA")

            vxc1    = _vxc_partial_deriv(rho, exc, vxc, fxc, spin, xctype="LDA")
            vrho1   = vxc1

            exc_jvp  = exc1 *  rho_t

            vrho_jvp = vrho1 * rho_t
            vxc_jvp  = (vrho_jvp, None, None, None)

        elif spin == 1:
            rho_t_u = rho_t[0]
            rho_t_d = rho_t[1]

            exc1    = _exc_partial_deriv(rho, exc, vxc, spin, "LDA")
            exc1_u, exc1_d = exc1

            vxc1    = _vxc_partial_deriv(rho, exc, vxc, fxc, spin, "LDA")
            vrho1   = vxc1
            vrho1_uu, vrho1_ud, vrho1_dd = vrho1
            
            exc_jvp = (exc1_u)/2 *  rho_t_u + (exc1_d)/2 *  rho_t_d

            vrho_jvp_u = vrho1_uu/2 * rho_t_u + vrho1_ud/2 * rho_t_d
            vrho_jvp_d = vrho1_dd/2 * rho_t_d + vrho1_ud/2 * rho_t_u
            vrho_jvp   = jnp.vstack((vrho_jvp_u, vrho_jvp_d)).T
            vxc_jvp    = (vrho_jvp, None, None, None)

        else:
            raise RuntimeError(f"spin = {spin}")

    elif fn_is_meta_gga:
        if spin == 0:
            exc1    = _exc_partial_deriv(rho, exc, vxc, spin, "MGGA")
            exc_jvp = jnp.einsum('np,np->p', exc1, rho_t)

            vxc1    = _vxc_partial_deriv(rho, exc, vxc, fxc, spin, "MGGA")
            vrho1, vsigma1, vlapl1, vtau1 = vxc1

            vrho_jvp   = jnp.einsum('np,np->p', vrho1,   rho_t)
            vsigma_jvp = jnp.einsum('np,np->p', vsigma1, rho_t)
            vlapl_jvp  = jnp.einsum('np,np->p', vlapl1,  rho_t)
            vtau_jvp   = jnp.einsum('np,np->p', vtau1,   rho_t)
            
            vrho1 = vsigma1 = vlapl1 = vtau1 = None
            vxc_jvp = jnp.vstack((vrho_jvp, vsigma_jvp, vlapl_jvp, vtau_jvp))

        elif spin == 1:
            rho_t_u = rho_t[0]
            rho_t_d = rho_t[1]

            exc1    = _exc_partial_deriv(rho, exc, vxc, spin, "MGGA")
            exc1_u, exc1_d = exc1

            vxc1    = _vxc_partial_deriv(rho, exc, vxc, fxc, spin, "MGGA")
            vrho1, vsigma1, vlapl1, vtau1      = vxc1
            vrho1_u,     vrho1_d = vrho1
            vlapl1_u,     vlapl1_d = vlapl1
            vtau1_uu,     vtau1_ud,   vtau1_dd = vtau1


            vsigma1_uu, vsigma1_ud, vsigma1_dd = vsigma1
            
            

            vrho_jvp_u   = jnp.einsum('np,np->p', vrho1_uu,   rho_t_u) + jnp.einsum('np,np->p', vrho1_ud,   rho_t_d)
            vsigma_jvp_u = jnp.einsum('np,np->p', vsigma1_uu, rho_t_u) + jnp.einsum('np,np->p', vsigma1_ud,   rho_t_d)
            vlapl_jvp_u  = jnp.einsum('np,np->p', vsigma1_uu, rho_t_u) + jnp.einsum('np,np->p', vsigma1_ud,   rho_t_d)
            vtau_jvp_u   = jnp.einsum('np,np->p', vsigma1_uu, rho_t_u) + jnp.einsum('np,np->p', vsigma1_ud,   rho_t_d)

            vrho_jvp   = jnp.einsum('np,np->p', vrho1,   rho_t)
            vsigma_jvp = jnp.einsum('np,np->p', vsigma1, rho_t)
            vlapl_jvp  = jnp.einsum('np,np->p', vlapl1,  rho_t)
            vtau_jvp   = jnp.einsum('np,np->p', vtau1,   rho_t)

        else:
            raise NotImplementedError

    elif fn_is_gga:
        if spin == 0:
            exc1    = _exc_partial_deriv(rho, exc, vxc, spin, "GGA")
            exc_jvp = jnp.einsum('np,np->p', exc1, rho_t)

            vxc1    = _vxc_partial_deriv(rho, exc, vxc, fxc, spin, "GGA")
            vrho1, vsigma1, _, _ = vxc1

            vrho_jvp   = jnp.einsum('np,np->p', vrho1,   rho_t)
            vsigma_jvp = jnp.einsum('np,np->p', vsigma1, rho_t)

            vrho1 = vsigma1 = None
            vxc_jvp = (vrho_jvp, vsigma_jvp, None, None)
        else:
            raise NotImplementedError

    else:
        raise RuntimeError

    if deriv == 0:
        vxc = fxc = kxc = vxc_jvp = fxc_jvp = kxc_jvp = None
    elif deriv == 1:
        fxc = kxc = fxc_jvp = kxc_jvp = None
    elif deriv == 2:
        kxc = kxc_jvp = None

    return (exc, vxc, fxc, kxc), (exc_jvp, vxc_jvp, fxc_jvp, kxc_jvp)

def _exc_partial_deriv(rho, exc, vxc, spin, xctype="LDA"):
    if xctype == "LDA":
        if   spin == 0:
            exc1  = (vxc[0] - exc) / rho
        elif spin == 1:
            rho_u = rho[0]
            rho_d = rho[1]

            vxc_u = vxc[0][:, 0]
            vxc_d = vxc[0][:, 1]

            exc_u = (vxc_u - exc)/rho_u/2
            exc_d = (vxc_d - exc)/rho_d/2

            exc   = (exc_u, exc_d)
        else:
            raise RuntimeError(f"spin = {spin}")

    elif xctype in ["GGA", "MGGA"]:
        if   spin == 0:
            exc1      = numpy.empty(rho.shape, dtype=rho.dtype)
            exc1[0]   = (vxc[0] - exc) / rho[0]
            exc1[1:4] = vxc[1] / rho[0] * 2. * rho[1:4]
            if xctype == "MGGA":
                exc1[4] = vxc[2] / rho[0]
                exc1[5] = vxc[3] / rho[0]

        elif spin == 1:
            exc1      = numpy.empty(rho.shape, dtype=rho.dtype)
            exc1[0]   = (vxc[0] - exc) / rho[0]
            exc1[1:4] = vxc[1] / rho[0] * 2. * rho[1:4]
            if xctype == "MGGA":
                exc1[4] = vxc[2] / rho[0]
                exc1[5] = vxc[3] / rho[0]

        else:
            raise RuntimeError(f"spin = {spin}")

    else:
        raise RuntimeError(f"xctype = {xctype}")

    return exc1

def _vxc_partial_deriv(rho, exc, vxc, fxc, spin, xctype="LDA"):
    vrho1 = vsigma1 = vlapl1 = vtau1 = None
    if xctype == "LDA":
        vrho1 = fxc[0]
    elif xctype in ["GGA", "MGGA"]:
        vrho1 = numpy.empty(rho.shape, dtype=rho.dtype)
        vrho1[0] = fxc[0]
        vrho1[1:4] = fxc[1] * 2. * rho[1:4]

        vsigma1 = numpy.empty(rho.shape, dtype=rho.dtype)
        vsigma1[0] = fxc[1]
        vsigma1[1:4] = fxc[2] * 2. * rho[1:4]

        if xctype == "MGGA":
            vrho1[4] = fxc[5]
            vrho1[5] = fxc[6]

            vsigma1[4] = fxc[8]
            vsigma1[5] = fxc[9]

            vlapl1 = numpy.empty(rho.shape, dtype=rho.dtype)
            vlapl1[0] = fxc[5]
            vlapl1[1:4] = fxc[8] * 2. * rho[1:4]
            vlapl1[4] = fxc[3]
            vlapl1[5] = fxc[7]

            vtau1 = numpy.empty(rho.shape, dtype=rho.dtype)
            vtau1[0] = fxc[6]
            vtau1[1:4] = fxc[9] * 2. * rho[1:4]
            vtau1[4] = fxc[7]
            vtau1[5] = fxc[4]
    else:
        raise KeyError
    return vrho1, vsigma1, vlapl1, vtau1
