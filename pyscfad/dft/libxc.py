from functools import partial
from pyscf.dft import libxc
from pyscf.dft.libxc import parse_xc, is_lda, is_meta_gga
from pyscfad import numpy as np
from pyscfad.ops import jit, custom_jvp

def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    # NOTE only consider exc and vxc
    if deriv > 1:
        raise NotImplementedError

    exc = _eval_xc_comp(rho, xc_code, spin, relativity, deriv=0, omega=omega, verbose=verbose)
    if deriv == 0:
        if spin == 0:
            vxc = (None,) * 4
        elif spin == 1:
            vxc = (None,) * 9
    elif deriv == 1:
        vxc = _eval_xc_comp(rho, xc_code, spin, relativity, deriv=1, omega=omega, verbose=verbose)
    return exc, vxc, None, None

@partial(custom_jvp, nondiff_argnums=tuple(range(1,7)))
def _eval_xc_comp(rho, xc_code, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    out = libxc.eval_xc(xc_code, rho, spin, relativity, deriv, omega, verbose)[deriv]
    if deriv == 1:
        out = tuple(out)
        if spin == 0:
            if len(out) < 4:
                out = out + (None,) * (4-len(out))
        elif spin == 1:
            if len(out) < 9:
                out = out + (None,) * (9-len(out))
    elif deriv == 2:
        out = tuple(out)
        if spin == 0:
            if len(out) < 10:
                out = out + (None,) * (10-len(out))
        elif spin == 1:
            if len(out) < 45:
                out = out + (None,) * (45-len(out))
    return out

@_eval_xc_comp.defjvp
def _eval_xc_comp_jvp(xc_code, spin, relativity, deriv, omega, verbose,
                      primals, tangents):
    rho, = primals
    rho_t, = tangents
    if deriv > 2:
        raise NotImplementedError

    val  = _eval_xc_comp(rho, xc_code, spin, relativity, deriv, omega, verbose)
    val1 = _eval_xc_comp(rho, xc_code, spin, relativity, deriv+1, omega, verbose)

    hyb, fn_facs = parse_xc(xc_code)
    fn_ids = [x[0] for x in fn_facs]
    n = len(fn_ids)
    if (n == 0 or
        all((is_lda(x) for x in fn_ids))):
        if spin == 0:
            if deriv == 0:
                exc1 = _exc_partial_deriv(rho, val, val1, 'LDA')
                jvp = exc1 * rho_t
            elif deriv == 1:
                vrho1 = _vxc_partial_deriv(rho, val, val1, 'LDA')[0]
                jvp = (vrho1 * rho_t,) + (None,) * 3
            else:
                v2rho2 = _fxc_partial_deriv(rho, val, val1, 'LDA')[0]
                jvp = (v2rho2 * rho_t,) + (None,) * 9
        elif spin == 1:
            rho_t_u = rho_t[0]
            rho_t_d = rho_t[1]
            if deriv == 0:
                exc1 = _exc_partial_deriv_polarized(rho, val, val1, 'LDA')
                exc1_u  = exc1[0]
                exc1_d  = exc1[1]
                jvp = exc1_u * rho_t_u + exc1_d * rho_t_d
            elif deriv == 1:
                vrho1 = _vxc_partial_deriv_polarized(rho, val, val1, 'LDA')[0]
                vrho1_u_u = vrho1[0]
                vrho1_u_d = vrho1[1]
                vrho1_d_d = vrho1[2]

                vrho_jvp_u = vrho1_u_u * rho_t_u + vrho1_u_d * rho_t_d
                vrho_jvp_d = vrho1_d_d * rho_t_d + vrho1_u_d * rho_t_u
                vrho_jvp   = np.vstack((vrho_jvp_u, vrho_jvp_d)).T
                jvp = (vrho_jvp,) + (None,) * 8
            else:
                raise NotImplementedError
    elif any((is_meta_gga(x) for x in fn_ids)):
        if spin == 0:
            if deriv == 0:
                exc1 = _exc_partial_deriv(rho, val, val1, 'MGGA')
                jvp = np.einsum('np,np->p', exc1, rho_t)
            elif deriv == 1:
                vrho1, vsigma1, vlapl1, vtau1 = _vxc_partial_deriv(rho, val, val1, 'MGGA')
                vrho_jvp = np.einsum('np,np->p', vrho1, rho_t)
                vsigma_jvp = np.einsum('np,np->p', vsigma1, rho_t)
                if vlapl1 is None:
                    vlapl_jvp = None
                else:
                    vlapl_jvp = np.einsum('np,np->p', vlapl1, rho_t)
                vtau_jvp = np.einsum('np,np->p', vtau1, rho_t)
                vrho1 = vsigma1 = vlapl1 = vtau1 = None
                jvp = (vrho_jvp, vsigma_jvp, vlapl_jvp, vtau_jvp)
            else:
                raise NotImplementedError
        elif spin == 1:
            rho_t_u = rho_t[0]
            rho_t_d = rho_t[1]
            if deriv == 0:
                exc1 = _exc_partial_deriv_polarized(rho, val, val1, 'MGGA')
                exc1_u, exc1_d = exc1
                jvp = (np.einsum('np,np->p', exc1_u, rho_t_u) +
                       np.einsum('np,np->p', exc1_d, rho_t_d))
            elif deriv == 1:
                raise NotImplementedError
#                vxc1 = _vxc_partial_deriv_polarized(rho, val, val1, 'MGGA')
#                vrho1, vsigma1, vlapl1, vtau1 = vxc1
#
#                vrho1_u_u = vrho1[0]
#                vrho1_u_d = vrho1[1]
#                vrho1_d_u = vrho1[2]
#                vrho1_d_d = vrho1[3]
#
#                vrho_jvp_u  = np.einsum('np,np->p', vrho1_u_u, rho_t_u)
#                vrho_jvp_u += np.einsum('np,np->p', vrho1_u_d, rho_t_d)
#                vrho_jvp_d  = np.einsum('np,np->p', vrho1_d_u, rho_t_u)
#                vrho_jvp_d += np.einsum('np,np->p', vrho1_d_d, rho_t_d)
#                vrho_jvp    = np.vstack((vrho_jvp_u, vrho_jvp_d)).T
#
#                vsigma1_uu_u = vsigma1[0]
#                vsigma1_uu_d = vsigma1[1]
#                vsigma1_ud_u = vsigma1[2]
#                vsigma1_ud_d = vsigma1[3]
#                vsigma1_dd_u = vsigma1[4]
#                vsigma1_dd_d = vsigma1[5]
#
#                vsigma_jvp_uu  = np.einsum('np,np->p', vsigma1_uu_u, rho_t_u)
#                vsigma_jvp_uu += np.einsum('np,np->p', vsigma1_uu_d, rho_t_d)
#                vsigma_jvp_ud  = np.einsum('np,np->p', vsigma1_ud_u, rho_t_u)
#                vsigma_jvp_ud += np.einsum('np,np->p', vsigma1_ud_d, rho_t_d)
#                vsigma_jvp_dd  = np.einsum('np,np->p', vsigma1_dd_u, rho_t_u)
#                vsigma_jvp_dd += np.einsum('np,np->p', vsigma1_dd_d, rho_t_d)
#                vsigma_jvp = np.vstack((vsigma_jvp_uu, vsigma_jvp_ud, vsigma_jvp_dd)).T
#
#                vlapl1_u_u = vlapl1[0]
#                vlapl1_u_d = vlapl1[1]
#                vlapl1_d_u = vlapl1[2]
#                vlapl1_d_d = vlapl1[3]
#
#                vlapl_jvp_u  = np.einsum('np,np->p', vlapl1_u_u, rho_t_u)
#                vlapl_jvp_u += np.einsum('np,np->p', vlapl1_u_d, rho_t_d)
#                vlapl_jvp_d  = np.einsum('np,np->p', vlapl1_d_d, rho_t_d)
#                vlapl_jvp_d += np.einsum('np,np->p', vlapl1_d_u, rho_t_u)
#                vlapl_jvp    = np.vstack((vlapl_jvp_u, vlapl_jvp_d)).T
#
#                vtau1_u_u = vtau1[0]
#                vtau1_u_d = vtau1[1]
#                vtau1_d_u = vtau1[2]
#                vtau1_d_d = vtau1[3]
#
#                vtau_jvp_u = (np.einsum('np,np->p', vtau1_u_u, rho_t_u) +
#                              np.einsum('np,np->p', vtau1_u_d, rho_t_d))
#                vtau_jvp_d = (np.einsum('np,np->p', vtau1_d_d, rho_t_d) +
#                              np.einsum('np,np->p', vtau1_d_u, rho_t_u))
#                vtau_jvp   = np.vstack((vtau_jvp_u, vtau_jvp_d)).T
#
#                jvp = (vrho_jvp, vsigma_jvp, vlapl_jvp, vtau_jvp) + (None,) * 5
            else:
                raise NotImplementedError
    else:
        if spin == 0:
            if deriv == 0:
                exc1 = _exc_partial_deriv(rho, val, val1, 'GGA')
                jvp = np.einsum('np,np->p', exc1, rho_t)
            elif deriv == 1:
                vrho1, vsigma1 = _vxc_partial_deriv(rho, val, val1, 'GGA')[:2]
                vrho_jvp = np.einsum('np,np->p', vrho1, rho_t)
                vsigma_jvp = np.einsum('np,np->p', vsigma1, rho_t)
                vrho1 = vsigma1 = None
                jvp = (vrho_jvp, vsigma_jvp, None, None)
            else:
                v2rho2, v2rhosigma, v2sigma2 = _fxc_partial_deriv(rho, val, val1, 'GGA')[:3]
                v2rho2_jvp = np.einsum('np,np->p', v2rho2, rho_t)
                v2rhosigma_jvp = np.einsum('np,np->p', v2rhosigma, rho_t)
                v2sigma2_jvp = np.einsum('np,np->p', v2sigma2, rho_t)
                jvp = (v2rho2_jvp, v2rhosigma_jvp, v2sigma2_jvp) + (None,)*7
        elif spin == 1:
            rho_t_u = rho_t[0]
            rho_t_d = rho_t[1]
            if deriv == 0:
                exc1 = _exc_partial_deriv_polarized(rho, val, val1, 'GGA')
                exc1_u, exc1_d = exc1
                jvp = (np.einsum('np,np->p', exc1_u, rho_t_u) +
                       np.einsum('np,np->p', exc1_d, rho_t_d))
            elif deriv == 1:
                vrho1, vsigma1 = _vxc_partial_deriv_polarized(rho, val, val1, 'GGA')[:2]

                vrho1_u_u = vrho1[0]
                vrho1_u_d = vrho1[1]
                vrho1_d_u = vrho1[2]
                vrho1_d_d = vrho1[3]

                vrho_jvp_u = (np.einsum('np,np->p', vrho1_u_u, rho_t_u) +
                              np.einsum('np,np->p', vrho1_u_d, rho_t_d))
                vrho_jvp_d = (np.einsum('np,np->p', vrho1_d_d, rho_t_d) +
                              np.einsum('np,np->p', vrho1_d_u, rho_t_u))
                vrho_jvp = np.vstack([vrho_jvp_u, vrho_jvp_d]).T

                vsigma1_uu_u = vsigma1[0]
                vsigma1_uu_d = vsigma1[1]
                vsigma1_ud_u = vsigma1[2]
                vsigma1_ud_d = vsigma1[3]
                vsigma1_dd_u = vsigma1[4]
                vsigma1_dd_d = vsigma1[5]

                vsigma_jvp_uu = (np.einsum('np,np->p', vsigma1_uu_u, rho_t_u) +
                                 np.einsum('np,np->p', vsigma1_uu_d, rho_t_d))
                vsigma_jvp_ud = (np.einsum('np,np->p', vsigma1_ud_u, rho_t_u) +
                                 np.einsum('np,np->p', vsigma1_ud_d, rho_t_d))
                vsigma_jvp_dd = (np.einsum('np,np->p', vsigma1_dd_u, rho_t_u) +
                                 np.einsum('np,np->p', vsigma1_dd_d, rho_t_d))
                vsigma_jvp = np.vstack([vsigma_jvp_uu, vsigma_jvp_ud, vsigma_jvp_dd]).T

                jvp = (vrho_jvp, vsigma_jvp) + (None,) * 7

            else:
                raise NotImplementedError

    return val, jvp

@partial(jit, static_argnames=['xctype'])
def _exc_partial_deriv(rho, exc, vxc, xctype='LDA'):
    if xctype == 'LDA':
        exc1 = (vxc[0] - exc) / rho
    elif xctype in ['GGA', 'MGGA']:
        drho = (vxc[0] - exc) / rho[0]
        dsigma = vxc[1] / rho[0] * 2. * rho[1:4]
        exc1 = np.vstack((drho, dsigma))
        if xctype == 'MGGA':
            if vxc[2] is None:
                dlap = np.zeros_like(rho[0])
            else:
                dlap = vxc[2] / rho[0]
            dtau = vxc[3] / rho[0]
            exc1 = np.vstack((exc1, dlap, dtau))
    else:
        raise KeyError

    return exc1

@partial(jit, static_argnames=['xctype'])
def _exc_partial_deriv_polarized(rho, exc, vxc, xctype='LDA'):
    if xctype == 'LDA':
        rho_tot = rho[0] + rho[1]

        exc1_rho1 = vxc[0]
        exc1_rho1_u = exc1_rho1[:, 0]
        exc1_rho1_d = exc1_rho1[:, 1]

        exc1_u = (exc1_rho1_u - exc) / rho_tot
        exc1_d = (exc1_rho1_d - exc) / rho_tot
        exc1   = (exc1_u, exc1_d)

    elif xctype in ['GGA', 'MGGA']:
        rho_u   = rho[0]
        rho_d   = rho[1]
        rho_tot = rho_u + rho_d

        exc1_rho1   = vxc[0]
        exc1_rho1_u = exc1_rho1[:, 0]
        exc1_rho1_d = exc1_rho1[:, 1]

        exc1_sigma1    = vxc[1]
        exc1_sigma1_uu = exc1_sigma1[:, 0]
        exc1_sigma1_ud = exc1_sigma1[:, 1]
        exc1_sigma1_dd = exc1_sigma1[:, 2]

        drho_u = (exc1_rho1_u - exc) / rho_tot[0]
        dsigma_u = (exc1_sigma1_uu / rho_tot[0] * rho_u[1:4] * 2 +
                    exc1_sigma1_ud / rho_tot[0] * rho_d[1:4])
        exc1_u = np.vstack([drho_u, dsigma_u])

        drho_d = (exc1_rho1_d - exc) / rho_tot[0]
        dsigma_d = (exc1_sigma1_dd / rho_tot[0] * rho_d[1:4] * 2 +
                    exc1_sigma1_ud / rho_tot[0] * rho_u[1:4])
        exc1_d = np.vstack([drho_d, dsigma_d])

        if xctype == 'MGGA':
            exc1_lapl1   = vxc[2]
            exc1_lapl1_u = exc1_lapl1[:, 0]
            exc1_lapl1_d = exc1_lapl1[:, 1]

            dlap_u = exc1_lapl1_u / rho_tot[0]
            dlap_d = exc1_lapl1_d / rho_tot[0]

            exc1_tau1   = vxc[3]
            exc1_tau1_u = exc1_tau1[:, 0]
            exc1_tau1_d = exc1_tau1[:, 1]

            dtau_u = exc1_tau1_u / rho_tot[0]
            dtau_d = exc1_tau1_d / rho_tot[0]

            exc1_u = np.vstack([exc1_u, dlap_u, dtau_u])
            exc1_d = np.vstack([exc1_d, dlap_d, dtau_d])

        exc1 = (exc1_u, exc1_d)

    else:
        raise KeyError

    return exc1

@partial(jit, static_argnames=['xctype'])
def _vxc_partial_deriv(rho, vxc, fxc, xctype='LDA'):
    vrho1 = vsigma1 = vlapl1 = vtau1 = None
    if xctype == 'LDA':
        vrho1 = fxc[0]
    elif xctype in ['GGA', 'MGGA']:
        vrho1 = np.vstack((fxc[0], fxc[1] * 2. * rho[1:4]))
        vsigma1 = np.vstack((fxc[1], fxc[2] * 2. * rho[1:4]))
        if xctype == 'MGGA':
            ZERO = np.zeros_like(rho[0])
            if vxc[2] is None:
                fxc3 = fxc5 = fxc7 = fxc8 = ZERO
            else:
                fxc3 = fxc[3]
                fxc5 = fxc[5]
                fxc7 = fxc[7]
                fxc8 = fxc[8]
            vrho1 = np.vstack((vrho1, fxc5, fxc[6]))
            vsigma1 = np.vstack((vsigma1, fxc8, fxc[9]))
            if vxc[2] is None:
                vlapl1 = None
            else:
                vlapl1 = np.vstack((fxc5, fxc8 * 2. * rho[1:4], fxc3, fxc7))
            vtau1 = np.vstack((fxc[6], fxc[9] * 2. * rho[1:4], fxc7, fxc[4]))
    else:
        raise KeyError
    return vrho1, vsigma1, vlapl1, vtau1

@partial(jit, static_argnames=['xctype'])
def _vxc_partial_deriv_polarized(rho, vxc, fxc, xctype='LDA'):
    vrho1 = vsigma1 = vlapl1 = vtau1 = None
    if xctype == 'LDA':
        exc2_rho2     = fxc[0]
        exc2_rho2_u_u = exc2_rho2[:, 0]
        exc2_rho2_u_d = exc2_rho2[:, 1]
        exc2_rho2_d_d = exc2_rho2[:, 2]
        vrho1 = (exc2_rho2_u_u, exc2_rho2_u_d, exc2_rho2_d_d)

    elif xctype in ['GGA', 'MGGA']:
        rho_u = rho[0]
        rho_d = rho[1]

        exc2_rho2 = fxc[0]
        exc2_rho2_u_u = exc2_rho2[:, 0]
        exc2_rho2_u_d = exc2_rho2[:, 1]
        exc2_rho2_d_d = exc2_rho2[:, 2]

        exc2_rhosigma = fxc[1]
        exc2_rhosigma_u_uu = exc2_rhosigma[:, 0]
        exc2_rhosigma_u_ud = exc2_rhosigma[:, 1]
        exc2_rhosigma_u_dd = exc2_rhosigma[:, 2]
        exc2_rhosigma_d_uu = exc2_rhosigma[:, 3]
        exc2_rhosigma_d_ud = exc2_rhosigma[:, 4]
        exc2_rhosigma_d_dd = exc2_rhosigma[:, 5]

        exc2_sigma2 = fxc[2]
        exc2_sigma2_uu_uu = exc2_sigma2[:, 0]
        exc2_sigma2_uu_ud = exc2_sigma2[:, 1]
        exc2_sigma2_uu_dd = exc2_sigma2[:, 2]
        exc2_sigma2_ud_ud = exc2_sigma2[:, 3]
        exc2_sigma2_ud_dd = exc2_sigma2[:, 4]
        exc2_sigma2_dd_dd = exc2_sigma2[:, 5]

        vrho1_u_u = np.vstack(
            [exc2_rho2_u_u,
             (exc2_rhosigma_u_uu * rho_u[1:4] * 2 +
              exc2_rhosigma_u_ud * rho_d[1:4])]
        )

        vrho1_u_d = np.vstack(
            [exc2_rho2_u_d,
             (exc2_rhosigma_u_dd * rho_d[1:4] * 2 +
              exc2_rhosigma_u_ud * rho_u[1:4])]
        )

        vrho1_d_u = np.vstack(
            [exc2_rho2_u_d,
             (exc2_rhosigma_d_uu * rho_u[1:4] * 2 +
              exc2_rhosigma_d_ud * rho_d[1:4])]
        )

        vrho1_d_d = np.vstack(
            [exc2_rho2_d_d,
             (exc2_rhosigma_d_dd * rho_d[1:4] * 2 +
              exc2_rhosigma_d_ud * rho_u[1:4])]
        )

        vsigma1_uu_u = np.vstack(
            [exc2_rhosigma_u_uu,
             (exc2_sigma2_uu_uu * rho_u[1:4] * 2 +
              exc2_sigma2_uu_ud * rho_d[1:4])]
        )

        vsigma1_uu_d = np.vstack(
            [exc2_rhosigma_d_uu,
             (exc2_sigma2_uu_dd * rho_d[1:4] * 2 +
              exc2_sigma2_uu_ud * rho_u[1:4])]
        )

        vsigma1_ud_u = np.vstack(
            [exc2_rhosigma_u_ud,
             (exc2_sigma2_uu_ud * rho_u[1:4] * 2 +
              exc2_sigma2_ud_ud * rho_d[1:4])]
        )

        vsigma1_ud_d = np.vstack(
            [exc2_rhosigma_d_ud,
             (exc2_sigma2_ud_dd * rho_d[1:4] * 2 +
              exc2_sigma2_ud_ud * rho_u[1:4])]
        )

        vsigma1_dd_u = np.vstack(
            [exc2_rhosigma_u_dd,
             (exc2_sigma2_uu_dd * rho_u[1:4] * 2 +
              exc2_sigma2_ud_dd * rho_d[1:4])]
        )

        vsigma1_dd_d = np.vstack(
            [exc2_rhosigma_d_dd,
             (exc2_sigma2_dd_dd * rho_d[1:4] * 2 +
              exc2_sigma2_ud_dd * rho_u[1:4])]
        )

        if xctype == 'MGGA':
            #FIXME: UKS-MGGA Gradient
            raise NotImplementedError

#            exc2_lapl2     = fxc[3]
#            exc2_tau2      = fxc[4]
#            exc2_rholapl   = fxc[5]
#            exc2_rhotau    = fxc[6]
#            exc2_lapltau   = fxc[7]
#            exc2_sigmalapl = fxc[8]
#            exc2_sigmatau  = fxc[9]
#
#            exc2_rholapl_u_u = exc2_rholapl[:, 0]
#            exc2_rholapl_u_d = exc2_rholapl[:, 1]
#            exc2_rholapl_d_d = exc2_rholapl[:, 2]
#            exc2_rholapl_d_u = exc2_rholapl[:, 3]
#
#            vrho1_u_u[4]     = exc2_rholapl_u_u
#            vrho1_u_d[4]     = exc2_rholapl_u_d
#            vrho1_d_u[4]     = exc2_rholapl_d_u
#            vrho1_d_d[4]     = exc2_rholapl_d_d
#
#            exc2_rhotau_u_u = exc2_rhotau[:, 0]
#            exc2_rhotau_u_d = exc2_rhotau[:, 1]
#            exc2_rhotau_d_u = exc2_rhotau[:, 2]
#            exc2_rhotau_d_d = exc2_rhotau[:, 3]
#
#            vrho1_u_u[5]     = exc2_rhotau_u_u
#            vrho1_u_d[5]     = exc2_rhotau_u_d
#            vrho1_d_u[5]     = exc2_rhotau_d_u
#            vrho1_d_d[5]     = exc2_rhotau_d_d
#
#            exc2_lapl2_u_u   = exc2_lapl2[:, 0]
#            exc2_lapl2_u_d   = exc2_lapl2[:, 1]
#            exc2_lapl2_d_d   = exc2_lapl2[:, 2]
#
#            exc2_sigmalapl_uu_u = exc2_sigmalapl[:, 0]
#            exc2_sigmalapl_ud_u = exc2_sigmalapl[:, 1]
#            exc2_sigmalapl_dd_u = exc2_sigmalapl[:, 2]
#            exc2_sigmalapl_uu_d = exc2_sigmalapl[:, 3]
#            exc2_sigmalapl_ud_d = exc2_sigmalapl[:, 4]
#            exc2_sigmalapl_dd_d = exc2_sigmalapl[:, 5]
#
#            exc2_lapltau_u_u    = exc2_lapltau[:, 0]
#            exc2_lapltau_u_d    = exc2_lapltau[:, 1]
#            exc2_lapltau_d_u    = exc2_lapltau[:, 2]
#            exc2_lapltau_d_d    = exc2_lapltau[:, 3]
#
#            vlapl1_u_u       = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
#            vlapl1_u_u[0]    = exc2_rholapl_u_u
#            vlapl1_u_u[1:4]  = exc2_sigmalapl_uu_u * rho_u[1:4] * 2
#            vlapl1_u_u[1:4] += exc2_sigmalapl_ud_u * rho_d[1:4]
#            vlapl1_u_u[4]    = exc2_lapl2_u_u
#            vlapl1_u_u[5]    = exc2_lapltau_u_u
#
#            vlapl1_u_d       = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
#            vlapl1_u_d[0]    = exc2_rholapl_d_u
#            vlapl1_u_d[1:4]  = exc2_sigmalapl_dd_u * rho_d[1:4] * 2
#            vlapl1_u_d[1:4] += exc2_sigmalapl_ud_u * rho_u[1:4]
#            vlapl1_u_d[4]    = exc2_lapl2_u_d
#            vlapl1_u_d[5]    = exc2_lapltau_u_d
#
#            vlapl1_d_u       = numpy.empty(rho_d.shape, dtype=rho_d.dtype)
#            vlapl1_d_u[0]    = exc2_rholapl_u_d
#            vlapl1_d_u[1:4]  = exc2_sigmalapl_uu_d * rho_u[1:4] * 2
#            vlapl1_d_u[1:4] += exc2_sigmalapl_ud_d * rho_d[1:4]
#            vlapl1_d_u[4]    = exc2_lapl2_u_d
#            vlapl1_d_u[5]    = exc2_lapltau_d_u
#
#            vlapl1_d_d       = numpy.empty(rho_d.shape, dtype=rho_d.dtype)
#            vlapl1_d_d[0]    = exc2_rholapl_d_d
#            vlapl1_d_d[1:4]  = exc2_sigmalapl_dd_d * rho_d[1:4] * 2
#            vlapl1_d_d[1:4] += exc2_sigmalapl_ud_d * rho_u[1:4]
#            vlapl1_d_d[4]    = exc2_lapl2_d_d
#            vlapl1_d_d[5]    = exc2_lapltau_d_d
#
#            vlapl1 = (vlapl1_u_u, vlapl1_u_d, vlapl1_d_u, vlapl1_d_d)
#
#            exc2_tau2_u_u      = exc2_tau2[:, 0]
#            exc2_tau2_u_d      = exc2_tau2[:, 1]
#            exc2_tau2_d_d      = exc2_tau2[:, 2]
#
#            exc2_sigmatau_uu_u = exc2_sigmatau[:, 0]
#            exc2_sigmatau_ud_u = exc2_sigmatau[:, 1]
#            exc2_sigmatau_dd_u = exc2_sigmatau[:, 2]
#            exc2_sigmatau_uu_d = exc2_sigmatau[:, 3]
#            exc2_sigmatau_ud_d = exc2_sigmatau[:, 4]
#            exc2_sigmatau_dd_d = exc2_sigmatau[:, 5]
#
#            vtau1_u_u          = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
#            vtau1_u_u[0]       = exc2_rhotau_u_u
#            vtau1_u_u[1:4]     = exc2_sigmatau_uu_u * rho_u[1:4] * 2
#            vtau1_u_u[1:4]    += exc2_sigmatau_ud_u * rho_d[1:4]
#            vtau1_u_u[4]       = exc2_lapltau_u_u
#            vtau1_u_u[5]       = exc2_tau2_u_u
#
#            vtau1_u_d          = numpy.empty(rho_d.shape, dtype=rho_d.dtype)
#            vtau1_u_d[0]       = exc2_rhotau_d_u
#            vtau1_u_d[1:4]     = exc2_sigmatau_dd_u * rho_d[1:4] * 2
#            vtau1_u_d[1:4]    += exc2_sigmatau_ud_u * rho_u[1:4]
#            vtau1_u_d[4]       = exc2_lapltau_d_u
#            vtau1_u_d[5]       = exc2_tau2_u_d
#
#            vtau1_d_u          = numpy.empty(rho_u.shape, dtype=rho_u.dtype)
#            vtau1_d_u[0]       = exc2_rhotau_u_d
#            vtau1_d_u[1:4]     = exc2_sigmatau_uu_d * rho_u[1:4] * 2
#            vtau1_d_u[1:4]    += exc2_sigmatau_ud_d * rho_d[1:4]
#            vtau1_d_u[4]       = exc2_lapltau_u_d
#            vtau1_d_u[5]       = exc2_tau2_u_d
#
#            vtau1_d_d          = numpy.empty(rho_d.shape, dtype=rho_d.dtype)
#            vtau1_d_d[0]       = exc2_rhotau_d_d
#            vtau1_d_d[1:4]     = exc2_sigmatau_dd_d * rho_d[1:4] * 2
#            vtau1_d_d[1:4]    += exc2_sigmatau_ud_d * rho_u[1:4]
#            vtau1_d_d[4]       = exc2_lapltau_d_d
#            vtau1_d_d[5]       = exc2_tau2_d_d
#
#            vtau1 = (vtau1_u_u, vtau1_u_d, vtau1_d_u, vtau1_d_d)

        vrho1 = (vrho1_u_u, vrho1_u_d, vrho1_d_u, vrho1_d_d)

        vsigma1 = (vsigma1_uu_u, vsigma1_uu_d,
                   vsigma1_ud_u, vsigma1_ud_d,
                   vsigma1_dd_u, vsigma1_dd_d)

    else:
        raise KeyError

    return vrho1, vsigma1, vlapl1, vtau1

@partial(jit, static_argnames=['xctype'])
def _fxc_partial_deriv(rho, fxc, kxc, xctype='LDA'):
    v2rho2_1 = v2rhosigma_1 = v2sigma2_1 = None
    if xctype == 'LDA':
        v2rho2_1 = kxc[0]
    elif xctype in ['GGA', 'MGGA']:
        v2rho2_1 = np.vstack((kxc[0], kxc[1] * 2. * rho[1:4]))
        v2rhosigma_1 = np.vstack((kxc[1], kxc[2] * 2. * rho[1:4]))
        v2sigma2_1 = np.vstack((kxc[2], kxc[3] * 2. * rho[1:4]))
        if xctype == 'MGGA':
            raise NotImplementedError
    else:
        raise KeyError
    return (v2rho2_1, v2rhosigma_1, v2sigma2_1) + (None,) * 7
