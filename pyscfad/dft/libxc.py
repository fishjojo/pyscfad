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
    r''' 
    Args:
        xc_code : str
            A string to describe the linear combination of different XC functionals.
            The X and C functional are separated by comma like '.8*LDA+.2*B86,VWN'.
            If "HF" (exact exchange) is appeared in the string, the HF part will
            be skipped.  If an empty string "" is given, the returns exc, vxc,...
            will be vectors of zeros.
        rho : ndarray
            Shape of ((*, ngrid)) for electron density (and derivatives) if spin = 0;
            Shape of ((*,ngrid), (*,ngrid)) for alpha/beta electron density (and derivatives) if spin > 0;

            rho (*,ngrid) are ordered as (den, grad_x, grad_y, grad_z, laplacian, tau)
            where grad_x = d/dx den, laplacian = \nabla^2 den, tau = 1/2(\nabla f)^2
            In spin unrestricted case,
            rho is ((den_u,grad_xu,grad_yu,grad_zu,laplacian_u,tau_u)
                    (den_d,grad_xd,grad_yd,grad_zd,laplacian_d,tau_d))

    Returns:
        epsilonxc, vxc, fxc, kxc

        where

        * epsilonxc = exc / den for restricted case

        * epsilonxc = exc / (den_u + den_d) for unrestricted case

        E_xc = \int dr exc(rho) = \int dr den_tot epsilonxc(rho)

        * vxc = (vrho, vsigma, vlapl, vtau) for restricted case

        vrho   = exc1_rho1 = \frac{\partial exc}{\partial rho}

        vsigma = exc1_sigma1 = \frac{\partial exc}{\partial sigma} in which sigma = \sum_{i = x, y, z}grad_i * grad_i

        vlapl  = exc1_lapl1 = \frac{\partial exc}{\partial laplacian}

        vtau   = exc1_tau1 = \frac{\partial exc}{\partial tau}

        * vxc for unrestricted case
          | exc1_rho1[:,2]   = (u, d)
          | exc1_sigma1[:,3] = (uu, ud, dd)
          | exc1_lapl1[:,2]  = (u, d)
          | exc1_tau1[:,2]   = (u, d)

        * fxc for restricted case:
          (exc2_rho2, exc2_rhosigma, exc2_sigma2, exc2_lapl2, 
           exc2_tau2, exc2_rholapl,  exc2_rhotau, exc2_lapltau,
           exc2_sigmalapl, exc2_sigmatau)

        * fxc for unrestricted case:
          | exc2_rho2[:,3]     = (u_u, u_d, d_d)
          | exc2_rhosigma[:,6] = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
          | exc2_sigma2[:,6]   = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)
          | exc2_lapl2[:,3]
          | exct_au2[:,3]
          | exc2_rholapl[:,4]
          | exc2_rhotau[:,4]
          | exc2_lapltau[:,4]
          | exc2_sigmalapl[:,6]
          | exc2_sigmatau[:,6]

        * kxc ...

        see also libxc_itrf.c
    '''

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

    if deriv >= 2:
        raise NotImplementedError

    epsilonxc, vxc, fxc, kxc = _eval_xc(rho, hyb, fn_facs, spin, relativity, deriv+1, verbose)
    exc_jvp = vxc_jvp = fxc_jvp = kxc_jvp = None

    fn_ids = [x[0] for x in fn_facs]
    n      = len(fn_ids)

    fn_is_lda      = (n == 0) or all((is_lda(x) for x in fn_ids))
    fn_is_meta_gga = any((is_meta_gga(x) for x in fn_ids))
    fn_is_gga      = (not fn_is_lda) and (not fn_is_meta_gga)

    if fn_is_lda:
        if spin == 0:
            exc1     = _exc_partial_deriv(rho, epsilonxc, vxc, spin, xctype="LDA")
            exc_jvp  = exc1  * rho_t

            vxc1     = _vxc_partial_deriv(rho, epsilonxc, vxc, fxc, spin, xctype="LDA")
            vrho1, vsigma1, vlapl1, vtau1 = vxc1

            vrho_jvp = vrho1 * rho_t

            vxc_jvp  = (vrho_jvp, None, None, None)

        elif spin == 1:
            rho_t_u = rho_t[0]
            rho_t_d = rho_t[1]

            exc1    = _exc_partial_deriv(rho, epsilonxc, vxc, spin, "LDA")
            exc1_u  = exc1[0]
            exc1_d  = exc1[1]
            exc_jvp = exc1_u * rho_t_u + exc1_d * rho_t_d

            vxc1    = _vxc_partial_deriv(rho, epsilonxc, vxc, fxc, spin, "LDA")
            vrho1, vsigma1, vlapl1, vtau1 = vxc1

            vrho1_u_u = vrho1[0]
            vrho1_u_d = vrho1[1]
            vrho1_d_u = vrho1[2]
            vrho1_d_d = vrho1[3]
            
            vrho_jvp_u  = vrho1_u_u * rho_t_u 
            vrho_jvp_u += vrho1_u_d * rho_t_d
            vrho_jvp_d  = vrho1_d_d * rho_t_d 
            vrho_jvp_d += vrho1_d_u * rho_t_u
            vrho_jvp    = jnp.vstack((vrho_jvp_u, vrho_jvp_d)).T  

            vxc_jvp     = (vrho_jvp, None, None, None)

    elif fn_is_gga:
        if spin == 0:
            exc1    = _exc_partial_deriv(rho, epsilonxc, vxc, spin, "GGA")
            exc_jvp = jnp.einsum('np,np->p', exc1,    rho_t)
            
            vxc1    = _vxc_partial_deriv(rho, epsilonxc, vxc, fxc, spin, "GGA")
            vrho1, vsigma1, vlapl1, vtau1 = vxc1

            vrho_jvp   = jnp.einsum('np,np->p', vrho1,   rho_t)
            vsigma_jvp = jnp.einsum('np,np->p', vsigma1, rho_t)
            vxc_jvp    = (vrho_jvp, vsigma_jvp, None, None)

        else:
            rho_t_u = rho_t[0]
            rho_t_d = rho_t[1]

            exc1    = _exc_partial_deriv(rho, epsilonxc, vxc, spin, "GGA")
            exc1_u, exc1_d = exc1
            exc_jvp        = jnp.einsum('np,np->p', exc1_u, rho_t_u) + jnp.einsum('np,np->p', exc1_d, rho_t_d)  

            vxc1    = _vxc_partial_deriv(rho, epsilonxc, vxc, fxc, spin, "GGA")
            vrho1, vsigma1, vlapl1, vtau1 = vxc1

            vrho1_u_u = vrho1[0]
            vrho1_u_d = vrho1[1]
            vrho1_d_u = vrho1[2]
            vrho1_d_d = vrho1[3]

            vrho_jvp_u = jnp.einsum('np,np->p', vrho1_u_u, rho_t_u) + jnp.einsum('np,np->p', vrho1_u_d, rho_t_d)
            vrho_jvp_d = jnp.einsum('np,np->p', vrho1_d_d, rho_t_d) + jnp.einsum('np,np->p', vrho1_d_u, rho_t_u)
            vrho_jvp   = jnp.vstack((vrho_jvp_u, vrho_jvp_d)).T  

            vsigma1_uu_u = vsigma1[0]
            vsigma1_uu_d = vsigma1[1]
            vsigma1_ud_u = vsigma1[2]
            vsigma1_ud_d = vsigma1[3]
            vsigma1_dd_u = vsigma1[4]
            vsigma1_dd_d = vsigma1[5]

            vsigma_jvp_uu = jnp.einsum('np,np->p', vsigma1_uu_u, rho_t_u) + jnp.einsum('np,np->p', vsigma1_uu_d, rho_t_d)
            vsigma_jvp_ud = jnp.einsum('np,np->p', vsigma1_ud_u, rho_t_u) + jnp.einsum('np,np->p', vsigma1_ud_d, rho_t_d)
            vsigma_jvp_dd = jnp.einsum('np,np->p', vsigma1_dd_u, rho_t_u) + jnp.einsum('np,np->p', vsigma1_dd_d, rho_t_d)
            vsigma_jvp   = jnp.vstack((vsigma_jvp_uu, vsigma_jvp_ud, vsigma_jvp_dd)).T  

            vxc_jvp    = (vrho_jvp, vsigma_jvp, None, None)

        vrho1 = vsigma1 = None

    elif fn_is_meta_gga:
        if spin == 0:
            exc1    = _exc_partial_deriv(rho, epsilonxc, vxc, spin, "MGGA")
            
            exc_jvp = jnp.einsum('np,np->p', exc1,    rho_t)
            
            vxc1    = _vxc_partial_deriv(rho, epsilonxc, vxc, fxc, spin, "MGGA")
            vrho1, vsigma1, vlapl1, vtau1 = vxc1

            vrho_jvp   = jnp.einsum('np,np->p', vrho1,   rho_t)
            vsigma_jvp = jnp.einsum('np,np->p', vsigma1, rho_t)
            vlapl_jvp  = jnp.einsum('np,np->p', vlapl1,  rho_t)
            vtau_jvp   = jnp.einsum('np,np->p', vtau1,   rho_t)

            vxc_jvp    = (vrho_jvp, vsigma_jvp, vlapl_jvp, vtau_jvp)

        else:
            rho_t_u = rho_t[0]
            rho_t_d = rho_t[1]

            exc1    = _exc_partial_deriv(rho, epsilonxc, vxc, spin, "MGGA")
            exc1_u, exc1_d = exc1

            exc_jvp        = jnp.einsum('np,np->p', exc1_u, rho_t_u) + jnp.einsum('np,np->p', exc1_d, rho_t_d)  

            vxc1    = _vxc_partial_deriv(rho, epsilonxc, vxc, fxc, spin, "MGGA")
            vrho1, vsigma1, vlapl1, vtau1 = vxc1

            vrho1_u_u = vrho1[0]
            vrho1_u_d = vrho1[1]
            vrho1_d_u = vrho1[2]
            vrho1_d_d = vrho1[3]

            vrho_jvp_u  = jnp.einsum('np,np->p', vrho1_u_u, rho_t_u) 
            vrho_jvp_u += jnp.einsum('np,np->p', vrho1_u_d, rho_t_d)
            vrho_jvp_d  = jnp.einsum('np,np->p', vrho1_d_u, rho_t_u) 
            vrho_jvp_d += jnp.einsum('np,np->p', vrho1_d_d, rho_t_d)
            vrho_jvp    = jnp.vstack((vrho_jvp_u, vrho_jvp_d)).T  

            vsigma1_uu_u = vsigma1[0]
            vsigma1_uu_d = vsigma1[1]
            vsigma1_ud_u = vsigma1[2]
            vsigma1_ud_d = vsigma1[3]
            vsigma1_dd_u = vsigma1[4]
            vsigma1_dd_d = vsigma1[5]

            vsigma_jvp_uu  = jnp.einsum('np,np->p', vsigma1_uu_u, rho_t_u) 
            vsigma_jvp_uu += jnp.einsum('np,np->p', vsigma1_uu_d, rho_t_d)
            vsigma_jvp_ud  = jnp.einsum('np,np->p', vsigma1_ud_u, rho_t_u) 
            vsigma_jvp_ud += jnp.einsum('np,np->p', vsigma1_ud_d, rho_t_d)
            vsigma_jvp_dd  = jnp.einsum('np,np->p', vsigma1_dd_u, rho_t_u) 
            vsigma_jvp_dd += jnp.einsum('np,np->p', vsigma1_dd_d, rho_t_d)
            vsigma_jvp   = jnp.vstack((vsigma_jvp_uu, vsigma_jvp_ud, vsigma_jvp_dd)).T  

            vlapl1_u_u = vlapl1[0]
            vlapl1_u_d = vlapl1[1]
            vlapl1_d_u = vlapl1[2]
            vlapl1_d_d = vlapl1[3]

            vlapl_jvp_u  = jnp.einsum('np,np->p', vlapl1_u_u, rho_t_u)
            vlapl_jvp_u += jnp.einsum('np,np->p', vlapl1_u_d, rho_t_d)
            vlapl_jvp_d  = jnp.einsum('np,np->p', vlapl1_d_d, rho_t_d) 
            vlapl_jvp_d += jnp.einsum('np,np->p', vlapl1_d_u, rho_t_u)
            vlapl_jvp    = jnp.vstack((vlapl_jvp_u, vlapl_jvp_d)).T 

            vtau1_u_u = vtau1[0]
            vtau1_u_d = vtau1[1]
            vtau1_d_u = vtau1[2]
            vtau1_d_d = vtau1[3]

            vtau_jvp_u = jnp.einsum('np,np->p', vtau1_u_u, rho_t_u) + jnp.einsum('np,np->p', vtau1_u_d, rho_t_d)
            vtau_jvp_d = jnp.einsum('np,np->p', vtau1_d_d, rho_t_d) + jnp.einsum('np,np->p', vtau1_d_u, rho_t_u)
            vtau_jvp   = jnp.vstack((vtau_jvp_u, vtau_jvp_d)).T 

            vxc_jvp    = (vrho_jvp, vsigma_jvp, vlapl_jvp, vtau_jvp)

    else:
        raise NotImplementedError

    if deriv == 0:
        vxc = fxc = kxc = vxc_jvp = fxc_jvp = kxc_jvp = None
    elif deriv == 1:
        fxc = kxc = fxc_jvp = kxc_jvp = None
    elif deriv == 2:
        kxc = kxc_jvp = None

    return (epsilonxc, vxc, fxc, kxc), (exc_jvp, vxc_jvp, fxc_jvp, kxc_jvp)

def _exc_partial_deriv(rho, epsilonxc, vxc, spin, xctype="LDA"):

    assert spin == 0 or spin == 1
    if xctype == "LDA":
        if   spin == 0: #TODO: should be a function
            exc1_rho1 = vxc[0]

            exc1  = (exc1_rho1 - epsilonxc) / rho

        elif spin == 1: #TODO: should be a function
            rho_u   = rho[0]
            rho_d   = rho[1]
            rho_tot = rho_u + rho_d

            exc1_rho1 = vxc[0]

            exc1_rho1_u = exc1_rho1[:, 0]
            exc1_rho1_d = exc1_rho1[:, 1]

            exc1_u = (exc1_rho1_u - epsilonxc) / rho_tot
            exc1_d = (exc1_rho1_d - epsilonxc) / rho_tot

            exc1   = (exc1_u, exc1_d)

    elif xctype in ["GGA", "MGGA"]:
        if   spin == 0: #TODO: should be a function
            exc1      = numpy.empty(rho.shape, dtype=rho.dtype)

            exc1_rho1   = vxc[0]
            exc1_sigma1 = vxc[1]

            exc1[0]   = (exc1_rho1 - epsilonxc) / rho[0]
            exc1[1:4] = exc1_sigma1 / rho[0] * 2. * rho[1:4]

            if xctype == "MGGA": #TODO: should be a function
                exc1_lapl1 = vxc[2]
                exc1_tau1  = vxc[3]
                exc1[4]    = exc1_lapl1 / rho[0]
                exc1[5]    = exc1_tau1  / rho[0]

        elif spin == 1: #TODO: should be a function
            rho_u   = rho[0]
            rho_d   = rho[1]
            rho_tot = rho_u + rho_d

            exc1_u = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
            exc1_d = numpy.empty(rho_d.shape, dtype=rho_d.dtype)

            exc1_rho1   = vxc[0]
            exc1_rho1_u = exc1_rho1[:, 0]
            exc1_rho1_d = exc1_rho1[:, 1]

            exc1_sigma1    = vxc[1]
            exc1_sigma1_uu = exc1_sigma1[:, 0]
            exc1_sigma1_ud = exc1_sigma1[:, 1]
            exc1_sigma1_dd = exc1_sigma1[:, 2]

            exc1_u[0]    = (exc1_rho1_u - epsilonxc) / rho_tot[0]
            exc1_u[1:4]  = exc1_sigma1_uu / rho_tot[0] * rho_u[1:4] * 2
            exc1_u[1:4] += exc1_sigma1_ud / rho_tot[0] * rho_d[1:4]
            exc1_d[0]    = (exc1_rho1_d - epsilonxc) / rho_tot[0]
            exc1_d[1:4]  = exc1_sigma1_dd / rho_tot[0] * rho_d[1:4] * 2 
            exc1_d[1:4] += exc1_sigma1_ud / rho_tot[0] * rho_u[1:4]

            if xctype == "MGGA": #TODO: should be a function
                exc1_lapl1   = vxc[2]
                exc1_lapl1_u = exc1_lapl1[:, 0]
                exc1_lapl1_d = exc1_lapl1[:, 1]

                exc1_u[4]  = exc1_lapl1_u / rho_tot[0]
                exc1_d[4]  = exc1_lapl1_d / rho_tot[0]

                exc1_tau1   = vxc[3]
                exc1_tau1_u = exc1_tau1[:, 0]
                exc1_tau1_d = exc1_tau1[:, 1]

                exc1_u[5]  = exc1_tau1_u / rho_tot[0]
                exc1_d[5]  = exc1_tau1_d / rho_tot[0]

            exc1 = (exc1_u, exc1_d)

    else:
        raise RuntimeError(f"xctype = {xctype}")

    return exc1

def _vxc_partial_deriv(rho, epsilonxc, vxc, fxc, spin, xctype="LDA"):
    vrho1 = vsigma1 = vlapl1 = vtau1 = None
    assert spin == 0 or spin == 1

    if xctype == "LDA":
        if   spin == 0: #TODO: should be a function
            vrho1 = fxc[0]

        elif spin == 1: #TODO: should be a function
            exc2_rho2     = fxc[0]
            exc2_rho2_u_u = exc2_rho2[:, 0]
            exc2_rho2_u_d = exc2_rho2[:, 1]
            exc2_rho2_d_d = exc2_rho2[:, 2]

            vrho1 = (exc2_rho2_u_u, exc2_rho2_u_d, exc2_rho2_u_d, exc2_rho2_d_d)

    elif xctype in ["GGA", "MGGA"]:
        if spin == 0: #TODO: should be a function
            exc2_rho2      = fxc[0]
            exc2_rhosigma  = fxc[1]
            exc2_sigma2    = fxc[2]

            vrho1 = numpy.empty(rho.shape, dtype=rho.dtype)
            vrho1[0]     = exc2_rho2
            vrho1[1:4]   = exc2_rhosigma * rho[1:4] * 2

            vsigma1 = numpy.empty(rho.shape, dtype=rho.dtype)
            vsigma1[0]   = exc2_rhosigma
            vsigma1[1:4] = exc2_sigma2 * rho[1:4] * 2

            if xctype == "MGGA": #TODO: should be a function
                exc2_lapl2     = fxc[3]
                exc2_tau2      = fxc[4]
                exc2_rholapl   = fxc[5]
                exc2_rhotau    = fxc[6]
                exc2_lapltau   = fxc[7]
                exc2_sigmalapl = fxc[8]
                exc2_sigmatau  = fxc[9]

                vrho1[4]     = exc2_rholapl
                vrho1[5]     = exc2_rhotau

                vsigma1[4]   = exc2_sigmalapl
                vsigma1[5]   = exc2_sigmatau

                vlapl1 = numpy.empty(rho.shape, dtype=rho.dtype)
                vlapl1[0]    = exc2_rholapl
                vlapl1[1:4]  = exc2_sigmalapl * 2. * rho[1:4]
                vlapl1[4]    = exc2_lapl2
                vlapl1[5]    = exc2_lapltau

                vtau1 = numpy.empty(rho.shape, dtype=rho.dtype)
                vtau1[0]     = exc2_rhotau
                vtau1[1:4]   = exc2_sigmatau * 2. * rho[1:4]
                vtau1[4]     = exc2_lapltau
                vtau1[5]     = exc2_tau2

        elif spin == 1: #TODO: should be a function
            rho_u = rho[0]
            rho_d = rho[1]

            exc2_rho2 = fxc[0]
            exc2_rho2_u_u      = exc2_rho2[:, 0]
            exc2_rho2_u_d      = exc2_rho2[:, 1]
            exc2_rho2_d_d      = exc2_rho2[:, 2]

            exc2_rhosigma = fxc[1]
            exc2_rhosigma_u_uu = exc2_rhosigma[:, 0]
            exc2_rhosigma_u_ud = exc2_rhosigma[:, 1]
            exc2_rhosigma_u_dd = exc2_rhosigma[:, 2]
            exc2_rhosigma_d_uu = exc2_rhosigma[:, 3]
            exc2_rhosigma_d_ud = exc2_rhosigma[:, 4]
            exc2_rhosigma_d_dd = exc2_rhosigma[:, 5]

            exc2_sigma2 = fxc[2]
            exc2_sigma2_uu_uu  = exc2_sigma2[:, 0]
            exc2_sigma2_uu_ud  = exc2_sigma2[:, 1]
            exc2_sigma2_uu_dd  = exc2_sigma2[:, 2]
            exc2_sigma2_ud_ud  = exc2_sigma2[:, 3]
            exc2_sigma2_ud_dd  = exc2_sigma2[:, 4]
            exc2_sigma2_dd_dd  = exc2_sigma2[:, 5]

            vrho1_u_u   = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
            vrho1_u_u[0]    = exc2_rho2_u_u
            vrho1_u_u[1:4]  = exc2_rhosigma_u_uu * rho_u[1:4] * 2
            vrho1_u_u[1:4] += exc2_rhosigma_u_ud * rho_d[1:4]

            vrho1_u_d   = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
            vrho1_u_d[0]    = exc2_rho2_u_d
            vrho1_u_d[1:4]  = exc2_rhosigma_u_dd * rho_d[1:4] * 2
            vrho1_u_d[1:4] += exc2_rhosigma_u_ud * rho_u[1:4]

            vrho1_d_u   = numpy.empty(rho_d.shape, dtype=rho_d.dtype)
            vrho1_d_u[0]    = exc2_rho2_u_d
            vrho1_d_u[1:4]  = exc2_rhosigma_d_uu * rho_u[1:4] * 2
            vrho1_d_u[1:4] += exc2_rhosigma_d_ud * rho_d[1:4]

            vrho1_d_d   = numpy.empty(rho_d.shape, dtype=rho_d.dtype)
            vrho1_d_d[0]    = exc2_rho2_d_d
            vrho1_d_d[1:4]  = exc2_rhosigma_d_dd * rho_d[1:4] * 2
            vrho1_d_d[1:4] += exc2_rhosigma_d_ud * rho_u[1:4]

            vsigma1_uu_u       = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
            vsigma1_uu_u[0]    = exc2_rhosigma_u_uu
            vsigma1_uu_u[1:4]  = exc2_sigma2_uu_uu * rho_u[1:4] * 2
            vsigma1_uu_u[1:4] += exc2_sigma2_uu_ud * rho_d[1:4]

            vsigma1_uu_d       = numpy.empty(rho_d.shape, dtype=rho_d.dtype)
            vsigma1_uu_d[0]    = exc2_rhosigma_d_uu
            vsigma1_uu_d[1:4]  = exc2_sigma2_uu_dd * rho_d[1:4] * 2
            vsigma1_uu_d[1:4] += exc2_sigma2_uu_ud * rho_u[1:4]

            vsigma1_ud_u       = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
            vsigma1_ud_u[0]    = exc2_rhosigma_u_ud
            vsigma1_ud_u[1:4]  = exc2_sigma2_uu_ud * rho_u[1:4] * 2
            vsigma1_ud_u[1:4] += exc2_sigma2_ud_ud * rho_d[1:4]

            vsigma1_ud_d       = numpy.empty(rho_d.shape, dtype=rho_d.dtype)
            vsigma1_ud_d[0]    = exc2_rhosigma_d_ud
            vsigma1_ud_d[1:4]  = exc2_sigma2_ud_dd * rho_d[1:4] * 2
            vsigma1_ud_d[1:4] += exc2_sigma2_ud_ud * rho_u[1:4]

            vsigma1_dd_u       = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
            vsigma1_dd_u[0]    = exc2_rhosigma_u_dd
            vsigma1_dd_u[1:4]  = exc2_sigma2_uu_dd * rho_u[1:4] * 2
            vsigma1_dd_u[1:4] += exc2_sigma2_ud_dd * rho_d[1:4]

            vsigma1_dd_d       = numpy.empty(rho_d.shape, dtype=rho_d.dtype)
            vsigma1_dd_d[0]    = exc2_rhosigma_d_dd
            vsigma1_dd_d[1:4]  = exc2_sigma2_dd_dd * rho_d[1:4] * 2
            vsigma1_dd_d[1:4] += exc2_sigma2_ud_dd * rho_u[1:4]

            if xctype == "MGGA": #TODO: should be a function
                #FIXME: UKS-MGGA Gradient
                raise NotImplemented

                exc2_lapl2     = fxc[3]
                exc2_tau2      = fxc[4]
                exc2_rholapl   = fxc[5]
                exc2_rhotau    = fxc[6]
                exc2_lapltau   = fxc[7]
                exc2_sigmalapl = fxc[8]
                exc2_sigmatau  = fxc[9]

                exc2_rholapl_u_u = exc2_rholapl[:, 0]
                exc2_rholapl_u_d = exc2_rholapl[:, 1]
                exc2_rholapl_d_d = exc2_rholapl[:, 2]
                exc2_rholapl_d_u = exc2_rholapl[:, 3]

                vrho1_u_u[4]     = exc2_rholapl_u_u
                vrho1_u_d[4]     = exc2_rholapl_u_d
                vrho1_d_u[4]     = exc2_rholapl_d_u
                vrho1_d_d[4]     = exc2_rholapl_d_d

                exc2_rhotau_u_u = exc2_rhotau[:, 0]
                exc2_rhotau_u_d = exc2_rhotau[:, 1]
                exc2_rhotau_d_u = exc2_rhotau[:, 2]
                exc2_rhotau_d_d = exc2_rhotau[:, 3]

                vrho1_u_u[5]     = exc2_rhotau_u_u
                vrho1_u_d[5]     = exc2_rhotau_u_d
                vrho1_d_u[5]     = exc2_rhotau_d_u
                vrho1_d_d[5]     = exc2_rhotau_d_d

                exc2_lapl2_u_u   = exc2_lapl2[:, 0]
                exc2_lapl2_u_d   = exc2_lapl2[:, 1]
                exc2_lapl2_d_d   = exc2_lapl2[:, 2]

                exc2_sigmalapl_uu_u = exc2_sigmalapl[:, 0]
                exc2_sigmalapl_ud_u = exc2_sigmalapl[:, 1]
                exc2_sigmalapl_dd_u = exc2_sigmalapl[:, 2]
                exc2_sigmalapl_uu_d = exc2_sigmalapl[:, 3]
                exc2_sigmalapl_ud_d = exc2_sigmalapl[:, 4]
                exc2_sigmalapl_dd_d = exc2_sigmalapl[:, 5]

                exc2_lapltau_u_u    = exc2_lapltau[:, 0]
                exc2_lapltau_u_d    = exc2_lapltau[:, 1]
                exc2_lapltau_d_u    = exc2_lapltau[:, 2]
                exc2_lapltau_d_d    = exc2_lapltau[:, 3]

                vlapl1_u_u       = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
                vlapl1_u_u[0]    = exc2_rholapl_u_u
                vlapl1_u_u[1:4]  = exc2_sigmalapl_uu_u * rho_u[1:4] * 2
                vlapl1_u_u[1:4] += exc2_sigmalapl_ud_u * rho_d[1:4]
                vlapl1_u_u[4]    = exc2_lapl2_u_u
                vlapl1_u_u[5]    = exc2_lapltau_u_u

                vlapl1_u_d       = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
                vlapl1_u_d[0]    = exc2_rholapl_d_u
                vlapl1_u_d[1:4]  = exc2_sigmalapl_dd_u * rho_d[1:4] * 2
                vlapl1_u_d[1:4] += exc2_sigmalapl_ud_u * rho_u[1:4]
                vlapl1_u_d[4]    = exc2_lapl2_u_d
                vlapl1_u_d[5]    = exc2_lapltau_u_d

                vlapl1_d_u       = numpy.empty(rho_d.shape, dtype=rho_d.dtype)
                vlapl1_d_u[0]    = exc2_rholapl_u_d
                vlapl1_d_u[1:4]  = exc2_sigmalapl_uu_d * rho_u[1:4] * 2
                vlapl1_d_u[1:4] += exc2_sigmalapl_ud_d * rho_d[1:4]
                vlapl1_d_u[4]    = exc2_lapl2_u_d
                vlapl1_d_u[5]    = exc2_lapltau_d_u

                vlapl1_d_d       = numpy.empty(rho_d.shape, dtype=rho_d.dtype)
                vlapl1_d_d[0]    = exc2_rholapl_d_d
                vlapl1_d_d[1:4]  = exc2_sigmalapl_dd_d * rho_d[1:4] * 2
                vlapl1_d_d[1:4] += exc2_sigmalapl_ud_d * rho_u[1:4]
                vlapl1_d_d[4]    = exc2_lapl2_d_d
                vlapl1_d_d[5]    = exc2_lapltau_d_d

                vlapl1 = (vlapl1_u_u, vlapl1_u_d, vlapl1_d_u, vlapl1_d_d)

                exc2_tau2_u_u      = exc2_tau2[:, 0]
                exc2_tau2_u_d      = exc2_tau2[:, 1]
                exc2_tau2_d_d      = exc2_tau2[:, 2]

                exc2_sigmatau_uu_u = exc2_sigmatau[:, 0]
                exc2_sigmatau_ud_u = exc2_sigmatau[:, 1]
                exc2_sigmatau_dd_u = exc2_sigmatau[:, 2]
                exc2_sigmatau_uu_d = exc2_sigmatau[:, 3]
                exc2_sigmatau_ud_d = exc2_sigmatau[:, 4]
                exc2_sigmatau_dd_d = exc2_sigmatau[:, 5]

                vtau1_u_u          = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
                vtau1_u_u[0]       = exc2_rhotau_u_u
                vtau1_u_u[1:4]     = exc2_sigmatau_uu_u * rho_u[1:4] * 2
                vtau1_u_u[1:4]    += exc2_sigmatau_ud_u * rho_d[1:4]
                vtau1_u_u[4]       = exc2_lapltau_u_u
                vtau1_u_u[5]       = exc2_tau2_u_u

                vtau1_u_d          = numpy.empty(rho_d.shape, dtype=rho_d.dtype)
                vtau1_u_d[0]       = exc2_rhotau_d_u
                vtau1_u_d[1:4]     = exc2_sigmatau_dd_u * rho_d[1:4] * 2
                vtau1_u_d[1:4]    += exc2_sigmatau_ud_u * rho_u[1:4]
                vtau1_u_d[4]       = exc2_lapltau_d_u
                vtau1_u_d[5]       = exc2_tau2_u_d

                vtau1_d_u          = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
                vtau1_d_u[0]       = exc2_rhotau_u_d
                vtau1_d_u[1:4]     = exc2_sigmatau_uu_d * rho_u[1:4] * 2
                vtau1_d_u[1:4]    += exc2_sigmatau_ud_d * rho_d[1:4]
                vtau1_d_u[4]       = exc2_lapltau_u_d
                vtau1_d_u[5]       = exc2_tau2_u_d

                vtau1_d_d          = numpy.empty(rho_d.shape, dtype=rho_d.dtype)
                vtau1_d_d[0]       = exc2_rhotau_d_d
                vtau1_d_d[1:4]     = exc2_sigmatau_dd_d * rho_d[1:4] * 2
                vtau1_d_d[1:4]    += exc2_sigmatau_ud_d * rho_u[1:4]
                vtau1_d_d[4]       = exc2_lapltau_d_d
                vtau1_d_d[5]       = exc2_tau2_d_d

                vtau1 = (vtau1_u_u, vtau1_u_d, vtau1_d_u, vtau1_d_d)
            
            vrho1 = (vrho1_u_u, vrho1_u_d, vrho1_d_u, vrho1_d_d)

            vsigma1 = (vsigma1_uu_u, vsigma1_uu_d, 
                       vsigma1_ud_u, vsigma1_ud_d,
                       vsigma1_dd_u, vsigma1_dd_d)

    else:
        raise KeyError

    return vrho1, vsigma1, vlapl1, vtau1
