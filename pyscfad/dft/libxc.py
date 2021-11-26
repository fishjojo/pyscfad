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

    v1_rho1 = v1_sigma1 = v1_lapl1 = v1_tau1 = None

    if fn_is_lda:
        if spin == 0:
            exc1    = _exc_partial_deriv(rho, exc, vxc, spin, xctype="LDA")

            vxc1    = _vxc_partial_deriv(rho, exc, vxc, fxc, spin, xctype="LDA")
            v1_rho1, v1_sigma1, v1_lapl1, v1_tau1 = vxc1

            exc_jvp  = exc1  * rho_t

            vrho_jvp = v1_rho1 * rho_t
            vxc_jvp  = (vrho_jvp, None, None, None)

        elif spin == 1:
            rho_t_u = rho_t[0]
            rho_t_d = rho_t[1]

            exc1    = _exc_partial_deriv(rho, exc, vxc, spin, "LDA")
            exc1_u, exc1_d = exc1

            vxc1    = _vxc_partial_deriv(rho, exc, vxc, fxc, spin, "LDA")
            v1_rho1, v1_sigma1, v1_lapl1, v1_tau1 = vxc1
            v1_rho1_uu, v1_rho1_ud, v1_rho1_dd    = v1_rho1
            
            exc_jvp    = exc1_u * rho_t_u + exc1_d * rho_t_d

            vrho_jvp_u = v1_rho1_uu * rho_t_u + v1_rho1_ud * rho_t_d
            vrho_jvp_d = v1_rho1_dd * rho_t_d + v1_rho1_ud * rho_t_u
            vrho_jvp   = jnp.vstack((vrho_jvp_u, vrho_jvp_d)).T            
            vxc_jvp    = (vrho_jvp, None, None, None)

        else:
            raise RuntimeError(f"spin = {spin}")

        v1_rho1 = v1_sigma1 = v1_lapl1 = v1_tau1 = None

    elif fn_is_meta_gga:
        raise NotImplementedError('meta-GGA')

    elif fn_is_gga:
        if spin == 0:
            print("rho_t = ", rho_t.shape)
            exc1    = _exc_partial_deriv(rho, exc, vxc, spin, "GGA")

            vxc1    = _vxc_partial_deriv(rho, exc, vxc, fxc, spin, "GGA")
            v1_rho1, v1_sigma1, v1_lapl1, v1_tau1 = vxc1

            exc_jvp    = jnp.einsum('np,np->p', exc1, rho_t)

            vrho_jvp   = jnp.einsum('np,np->p', v1_rho1,   rho_t)
            vsigma_jvp = jnp.einsum('np,np->p', v1_sigma1, rho_t)
            vxc_jvp    = (vrho_jvp, vsigma_jvp, None, None)

        else:
            rho_t_u = rho_t[0]
            rho_t_d = rho_t[1]

            exc1    = _exc_partial_deriv(rho, exc, vxc, spin, "LDA")
            exc1_u, exc1_d = exc1

            vxc1    = _vxc_partial_deriv(rho, exc, vxc, fxc, spin, "LDA")
            v1_rho1, v1_sigma1, v1_lapl1, v1_tau1    = vxc1
            v1_rho1_uu, v1_rho1_ud, v1_rho1_dd       = v1_rho1
            v1_sigma1_uu, v1_sigma1_ud, v1_sigma1_dd = v1_sigma1

            exc_jvp    = jnp.einsum('np,np->p', exc1_u, rho_t_u) + jnp.einsum('np,np->p', exc1_d, rho_t_d)  

            vrho_jvp_u = jnp.einsum('np,np->p', v1_rho1_uu, rho_t_u) + jnp.einsum('np,np->p', v1_rho1_ud, rho_t_d)
            vrho_jvp_d = jnp.einsum('np,np->p', v1_rho1_dd, rho_t_d) + jnp.einsum('np,np->p', v1_rho1_ud, rho_t_u)
            vrho_jvp   = jnp.vstack((vrho_jvp_u, vrho_jvp_d)).T  

            vsigma_jvp_u = jnp.einsum('np,np->p', v1_sigma1_uu, rho_t_u) + jnp.einsum('np,np->p', v1_sigma1_ud, rho_t_d)
            vsigma_jvp_d = jnp.einsum('np,np->p', v1_sigma1_dd, rho_t_d) + jnp.einsum('np,np->p', v1_sigma1_ud, rho_t_u)
            vsigma_jvp   = jnp.vstack((vsigma_jvp_u, vsigma_jvp_d)).T  

            vxc_jvp    = (vrho_jvp, vsigma_jvp, None, None)

        v1_rho1 = v1_sigma1 = None

    else:
        raise NotImplementedError

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

            v1_rho1 = vxc[0]

            v1_rho1_u = v1_rho1[:, 0]
            v1_rho1_d = v1_rho1[:, 1]

            exc_u = (v1_rho1_u - exc) / rho_u / 2
            exc_d = (v1_rho1_d - exc) / rho_d / 2

            exc1  = (exc_u, exc_d)
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
            rho_u = rho[0]
            rho_d = rho[1]

            exc_u = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
            exc_d = numpy.empty(rho_d.shape, dtype=rho_d.dtype)

            v1_rho1   = vxc[0]
            v1_rho1_u = v1_rho1[:, 0]
            v1_rho1_d = v1_rho1[:, 1]

            v1_sigma1 = vxc[1]
            v1_rho1_uu = v1_sigma1[:, 0]
            v1_rho1_ud = v1_sigma1[:, 1]
            v1_rho1_dd = v1_sigma1[:, 2]

            exc_u[0]   = (v1_rho1_u - exc) / rho_u[0] / 2 
            exc_u[1:4] = v1_rho1_uu / rho_u[0] * rho_u[1:4] + v1_rho1_ud / rho_d[0] * rho_d[1:4]
            exc_d[0]   = (v1_rho1_d - exc) / rho_d[0] / 2 
            exc_d[1:4] = v1_rho1_dd / rho_d[0] * rho_d[1:4] + v1_rho1_ud / rho_u[0] * rho_u[1:4]

            if xctype == "MGGA":
                v1_lapl1   = vxc[2]
                v1_lapl1_u = v1_lapl1[:, 0]
                v1_lapl1_d = v1_lapl1[:, 1]
                exc_u[4]   = v1_lapl1_u / rho_u[0] / 2 
                exc_d[4]   = v1_lapl1_d / rho_d[0] / 2 

                v1_tau1    = vxc[3]
                v1_tau1_u  = v1_tau1[:, 0]
                v1_tau1_d  = v1_tau1[:, 1]
                exc_u[4]   = v1_tau1_u / rho_u[0] / 2 
                exc_d[4]   = v1_tau1_d / rho_d[0] / 2 

            exc1       = (exc_u, exc_d)

        else:
            raise RuntimeError(f"spin = {spin}")

    else:
        raise RuntimeError(f"xctype = {xctype}")

    return exc1

def _vxc_partial_deriv(rho, exc, vxc, fxc, spin, xctype="LDA"):
    vrho1 = vsigma1 = vlapl1 = vtau1 = None
    if xctype == "LDA":
        if   spin == 0:
            vrho1 = fxc[0]

        elif spin == 1:
            fxc_uu = fxc[0][:, 0] / 2
            fxc_ud = fxc[0][:, 1] / 2
            fxc_dd = fxc[0][:, 2] / 2

            vrho1 = (fxc_uu, fxc_ud, fxc_dd)
        else:
            raise RuntimeError(f"spin = {spin}")

    elif xctype in ["GGA", "MGGA"]:
        if spin == 0:
            vrho1 = numpy.empty(rho.shape, dtype=rho.dtype)
            vrho1[0]   = fxc[0]
            vrho1[1:4] = fxc[1] * 2. * rho[1:4]

            vsigma1 = numpy.empty(rho.shape, dtype=rho.dtype)
            vsigma1[0]   = fxc[1]
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

        elif spin == 1:
            vrho1 = numpy.empty(rho.shape, dtype=rho.dtype)
            vrho1[0] = fxc[0]
            vrho1[1:4] = fxc[1] * 2. * rho[1:4]

            fxc_uu = fxc[0][:, 0] / 2
            fxc_ud = fxc[0][:, 1] / 2
            fxc_dd = fxc[0][:, 2] / 2

            vrho1 = (fxc_uu, fxc_ud, fxc_dd)

            vsigma1 = numpy.empty(rho.shape, dtype=rho.dtype)
            vsigma1[0] = fxc[1]
            vsigma1[1:4] = fxc[2] * 2. * rho[1:4]

    else:
        raise KeyError
    return vrho1, vsigma1, vlapl1, vtau1
