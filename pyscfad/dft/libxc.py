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
from pyscf.dft import libxc
from pyscf.dft.libxc import (
    parse_xc,
    xc_type,
    is_hybrid_xc,
    is_nlc,
    is_lda,
    is_meta_gga,
    eval_xc1,
    rsh_coeff,
    hybrid_coeff,
    nlc_coeff,
    __version__,
    __reference__,
) # pylint: disable=unused-import
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
                vrho2 = _fxc_partial_deriv(rho, val, val1, 'LDA')[0]
                jvp = (vrho2 * rho_t,) + (None,) * 9
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
                vrho1_uu = vrho1[0]
                vrho1_ud = vrho1[1]
                vrho1_dd = vrho1[2]

                vrho_jvp_u = vrho1_uu * rho_t_u + vrho1_ud * rho_t_d
                vrho_jvp_d = vrho1_dd * rho_t_d + vrho1_ud * rho_t_u
                vrho_jvp   = np.vstack((vrho_jvp_u, vrho_jvp_d)).T
                jvp = (vrho_jvp,) + (None,) * 8
            else:
                frho1 = _fxc_partial_deriv_polarized(rho, val, val1, 'LDA')[0]
                #pylint: disable=E1136
                frho1_uuu = frho1[0]
                frho1_uud = frho1[1]
                frho1_udd = frho1[2]
                frho1_ddd = frho1[3]

                frho_jvp_uu = frho1_uuu * rho_t_u + frho1_uud * rho_t_d
                frho_jvp_ud = frho1_uud * rho_t_u + frho1_udd * rho_t_d
                frho_jvp_dd = frho1_udd * rho_t_u + frho1_ddd * rho_t_d
                frho_jvp = np.vstack((frho_jvp_uu, frho_jvp_ud, frho_jvp_dd)).T
                jvp = (frho_jvp,) + (None,) * 44
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
                v2rho2_1, v2rhosigma_1, v2sigma2_1 = (
                    _fxc_partial_deriv_polarized(rho, val, val1, xctype='GGA')[:3])

                #pylint: disable=E1136
                v2rho2_1_uu_u = v2rho2_1[0]
                v2rho2_1_uu_d = v2rho2_1[1]
                v2rho2_1_ud_u = v2rho2_1[2]
                v2rho2_1_ud_d = v2rho2_1[3]
                v2rho2_1_dd_u = v2rho2_1[4]
                v2rho2_1_dd_d = v2rho2_1[5]

                frho_jvp_uu = (np.einsum('np,np->p', v2rho2_1_uu_u, rho_t_u) +
                               np.einsum('np,np->p', v2rho2_1_uu_d, rho_t_d))
                frho_jvp_ud = (np.einsum('np,np->p', v2rho2_1_ud_u, rho_t_u) +
                               np.einsum('np,np->p', v2rho2_1_ud_d, rho_t_d))
                frho_jvp_dd = (np.einsum('np,np->p', v2rho2_1_dd_u, rho_t_u) +
                               np.einsum('np,np->p', v2rho2_1_dd_d, rho_t_d))
                frho_jvp = np.vstack([frho_jvp_uu, frho_jvp_ud, frho_jvp_dd]).T

                #pylint: disable=W0632
                (v2rhosigma_1_u_uu_u, v2rhosigma_1_u_uu_d,
                 v2rhosigma_1_u_ud_u, v2rhosigma_1_u_ud_d,
                 v2rhosigma_1_u_dd_u, v2rhosigma_1_u_dd_d,
                 v2rhosigma_1_d_uu_u, v2rhosigma_1_d_uu_d,
                 v2rhosigma_1_d_ud_u, v2rhosigma_1_d_ud_d,
                 v2rhosigma_1_d_dd_u, v2rhosigma_1_d_dd_d,) = v2rhosigma_1

                frhosig_jvp_u_uu = (np.einsum('np,np->p', v2rhosigma_1_u_uu_u, rho_t_u) +
                                    np.einsum('np,np->p', v2rhosigma_1_u_uu_d, rho_t_d))

                frhosig_jvp_u_ud = (np.einsum('np,np->p', v2rhosigma_1_u_ud_u, rho_t_u) +
                                    np.einsum('np,np->p', v2rhosigma_1_u_ud_d, rho_t_d))

                frhosig_jvp_u_dd = (np.einsum('np,np->p', v2rhosigma_1_u_dd_u, rho_t_u) +
                                    np.einsum('np,np->p', v2rhosigma_1_u_dd_d, rho_t_d))

                frhosig_jvp_d_uu = (np.einsum('np,np->p', v2rhosigma_1_d_uu_u, rho_t_u) +
                                    np.einsum('np,np->p', v2rhosigma_1_d_uu_d, rho_t_d))

                frhosig_jvp_d_ud = (np.einsum('np,np->p', v2rhosigma_1_d_ud_u, rho_t_u) +
                                    np.einsum('np,np->p', v2rhosigma_1_d_ud_d, rho_t_d))

                frhosig_jvp_d_dd = (np.einsum('np,np->p', v2rhosigma_1_d_dd_u, rho_t_u) +
                                    np.einsum('np,np->p', v2rhosigma_1_d_dd_d, rho_t_d))

                frhosig_jvp = np.vstack(
                    [frhosig_jvp_u_uu, frhosig_jvp_u_ud, frhosig_jvp_u_dd,
                     frhosig_jvp_d_uu, frhosig_jvp_d_ud, frhosig_jvp_d_dd]
                ).T

                #pylint: disable=W0632
                (v2sigma2_1_uu_uu_u,
                 v2sigma2_1_uu_uu_d,
                 v2sigma2_1_uu_ud_u,
                 v2sigma2_1_uu_ud_d,
                 v2sigma2_1_uu_dd_u,
                 v2sigma2_1_uu_dd_d,
                 v2sigma2_1_ud_ud_u,
                 v2sigma2_1_ud_ud_d,
                 v2sigma2_1_ud_dd_u,
                 v2sigma2_1_ud_dd_d,
                 v2sigma2_1_dd_dd_u,
                 v2sigma2_1_dd_dd_d,) = v2sigma2_1

                fsigma_jvp_uu_uu = (np.einsum('np,np->p', v2sigma2_1_uu_uu_u, rho_t_u) +
                                    np.einsum('np,np->p', v2sigma2_1_uu_uu_d, rho_t_d))

                fsigma_jvp_uu_ud = (np.einsum('np,np->p', v2sigma2_1_uu_ud_u, rho_t_u) +
                                    np.einsum('np,np->p', v2sigma2_1_uu_ud_d, rho_t_d))

                fsigma_jvp_uu_dd = (np.einsum('np,np->p', v2sigma2_1_uu_dd_u, rho_t_u) +
                                    np.einsum('np,np->p', v2sigma2_1_uu_dd_d, rho_t_d))

                fsigma_jvp_ud_ud = (np.einsum('np,np->p', v2sigma2_1_ud_ud_u, rho_t_u) +
                                    np.einsum('np,np->p', v2sigma2_1_ud_ud_d, rho_t_d))

                fsigma_jvp_ud_dd = (np.einsum('np,np->p', v2sigma2_1_ud_dd_u, rho_t_u) +
                                    np.einsum('np,np->p', v2sigma2_1_ud_dd_d, rho_t_d))

                fsigma_jvp_dd_dd = (np.einsum('np,np->p', v2sigma2_1_dd_dd_u, rho_t_u) +
                                    np.einsum('np,np->p', v2sigma2_1_dd_dd_d, rho_t_d))

                fsigma_jvp = np.vstack(
                    [fsigma_jvp_uu_uu, fsigma_jvp_uu_ud, fsigma_jvp_uu_dd,
                     fsigma_jvp_ud_ud, fsigma_jvp_ud_dd, fsigma_jvp_ud_dd]
                ).T

                jvp = (frho_jvp, frhosig_jvp, fsigma_jvp) + (None,) * 42

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

@partial(jit, static_argnames=['xctype'])
def _fxc_partial_deriv_polarized(rho, fxc, kxc, xctype='LDA'):
    v2rho2_1 = v2rhosigma_1 = v2sigma2_1 = None

    v3rho3 = kxc[0]
    v3rho3_uuu = kxc[0][:,0]
    v3rho3_uud = kxc[0][:,1]
    v3rho3_udd = kxc[0][:,2]
    v3rho3_ddd = kxc[0][:,3]

    if xctype == 'LDA':
        v2rho2_1 = (v3rho3_uuu,
                    v3rho3_uud,
                    v3rho3_udd,
                    v3rho3_ddd)
    elif xctype in ['GGA', 'MGGA']:
        rho_u = rho[0]
        rho_d = rho[1]

        v3rho2sigma = kxc[1]
        v3rho2sigma_uu_uu = v3rho2sigma[:,0]
        v3rho2sigma_uu_ud = v3rho2sigma[:,1]
        v3rho2sigma_uu_dd = v3rho2sigma[:,2]
        v3rho2sigma_ud_uu = v3rho2sigma[:,3]
        v3rho2sigma_ud_ud = v3rho2sigma[:,4]
        v3rho2sigma_ud_dd = v3rho2sigma[:,5]
        v3rho2sigma_dd_uu = v3rho2sigma[:,6]
        v3rho2sigma_dd_ud = v3rho2sigma[:,7]
        v3rho2sigma_dd_dd = v3rho2sigma[:,8]

        v2rho2_1_uu_u = np.vstack(
            [v3rho3_uuu,
             (v3rho2sigma_uu_uu * rho_u[1:4] * 2 +
              v3rho2sigma_uu_ud * rho_d[1:4])]
        )

        v2rho2_1_uu_d = np.vstack(
            [v3rho3_uud,
             (v3rho2sigma_uu_dd * rho_d[1:4] * 2 +
              v3rho2sigma_uu_ud * rho_u[1:4])]
        )

        v2rho2_1_ud_u = np.vstack(
            [v3rho3_uud,
             (v3rho2sigma_ud_uu * rho_u[1:4] * 2 +
              v3rho2sigma_ud_ud * rho_d[1:4])]
        )

        v2rho2_1_ud_d = np.vstack(
            [v3rho3_udd,
             (v3rho2sigma_ud_dd * rho_d[1:4] * 2 +
              v3rho2sigma_ud_ud * rho_u[1:4])]
        )

        v2rho2_1_dd_u = np.vstack(
            [v3rho3_udd,
             (v3rho2sigma_dd_uu * rho_u[1:4] * 2 +
              v3rho2sigma_dd_ud * rho_d[1:4])]
        )

        v2rho2_1_dd_d = np.vstack(
            [v3rho3_ddd,
             (v3rho2sigma_dd_dd * rho_d[1:4] * 2 +
              v3rho2sigma_dd_ud * rho_u[1:4])]
        )

        v2rho2_1 = (v2rho2_1_uu_u, v2rho2_1_uu_d,
                    v2rho2_1_ud_u, v2rho2_1_ud_d,
                    v2rho2_1_dd_u, v2rho2_1_dd_d)

        v3rhosigma2 = kxc[2]
        v3rhosigma2_u_uu_uu = v3rhosigma2[:,0]
        v3rhosigma2_u_uu_ud = v3rhosigma2[:,1]
        v3rhosigma2_u_uu_dd = v3rhosigma2[:,2]
        v3rhosigma2_u_ud_ud = v3rhosigma2[:,3]
        v3rhosigma2_u_ud_dd = v3rhosigma2[:,4]
        v3rhosigma2_u_dd_dd = v3rhosigma2[:,5]
        v3rhosigma2_d_uu_uu = v3rhosigma2[:,6]
        v3rhosigma2_d_uu_ud = v3rhosigma2[:,7]
        v3rhosigma2_d_uu_dd = v3rhosigma2[:,8]
        v3rhosigma2_d_ud_ud = v3rhosigma2[:,9]
        v3rhosigma2_d_ud_dd = v3rhosigma2[:,10]
        v3rhosigma2_d_dd_dd = v3rhosigma2[:,11]

        v2rhosigma_1_u_uu_u = np.vstack(
            [v3rho2sigma_uu_uu,
             (v3rhosigma2_u_uu_uu * rho_u[1:4] * 2 +
              v3rhosigma2_u_uu_ud * rho_d[1:4])]
        )
        v2rhosigma_1_u_uu_d = np.vstack(
            [v3rho2sigma_ud_uu,
             (v3rhosigma2_u_uu_dd * rho_d[1:4] * 2 +
              v3rhosigma2_u_uu_ud * rho_u[1:4])]
        )
        v2rhosigma_1_u_ud_u = np.vstack(
            [v3rho2sigma_uu_ud,
             (v3rhosigma2_u_uu_ud * rho_u[1:4] * 2 +
              v3rhosigma2_u_ud_ud * rho_d[1:4])]
        )
        v2rhosigma_1_u_ud_d = np.vstack(
            [v3rho2sigma_ud_ud,
             (v3rhosigma2_u_ud_dd * rho_d[1:4] * 2 +
              v3rhosigma2_u_ud_ud * rho_u[1:4])]
        )
        v2rhosigma_1_u_dd_u = np.vstack(
            [v3rho2sigma_uu_dd,
             (v3rhosigma2_u_uu_dd * rho_u[1:4] * 2 +
              v3rhosigma2_u_ud_dd * rho_d[1:4])]
        )
        v2rhosigma_1_u_dd_d = np.vstack(
            [v3rho2sigma_ud_dd,
             (v3rhosigma2_u_dd_dd * rho_d[1:4] * 2 +
              v3rhosigma2_u_ud_dd * rho_u[1:4])]
        )

        v2rhosigma_1_d_uu_u = np.vstack(
            [v3rho2sigma_ud_uu,
             (v3rhosigma2_d_uu_uu * rho_u[1:4] * 2 +
              v3rhosigma2_d_uu_ud * rho_d[1:4])]
        )
        v2rhosigma_1_d_uu_d = np.vstack(
            [v3rho2sigma_dd_uu,
             (v3rhosigma2_d_uu_dd * rho_d[1:4] * 2 +
              v3rhosigma2_d_uu_ud * rho_u[1:4])]
        )
        v2rhosigma_1_d_ud_u = np.vstack(
            [v3rho2sigma_ud_ud,
             (v3rhosigma2_d_uu_ud * rho_u[1:4] * 2 +
              v3rhosigma2_d_ud_ud * rho_d[1:4])]
        )
        v2rhosigma_1_d_ud_d = np.vstack(
            [v3rho2sigma_dd_ud,
             (v3rhosigma2_d_ud_dd * rho_d[1:4] * 2 +
              v3rhosigma2_d_ud_ud * rho_u[1:4])]
        )
        v2rhosigma_1_d_dd_u = np.vstack(
            [v3rho2sigma_ud_dd,
             (v3rhosigma2_d_uu_dd * rho_u[1:4] * 2 +
              v3rhosigma2_d_ud_dd * rho_d[1:4])]
        )
        v2rhosigma_1_d_dd_d = np.vstack(
            [v3rho2sigma_dd_dd,
             (v3rhosigma2_d_dd_dd * rho_d[1:4] * 2 +
              v3rhosigma2_d_ud_dd * rho_u[1:4])]
        )

        v2rhosigma_1 = (
            v2rhosigma_1_u_uu_u, v2rhosigma_1_u_uu_d,
            v2rhosigma_1_u_ud_u, v2rhosigma_1_u_ud_d,
            v2rhosigma_1_u_dd_u, v2rhosigma_1_u_dd_d,
            v2rhosigma_1_d_uu_u, v2rhosigma_1_d_uu_d,
            v2rhosigma_1_d_ud_u, v2rhosigma_1_d_ud_d,
            v2rhosigma_1_d_dd_u, v2rhosigma_1_d_dd_d,
        )

        v3sigma3 = kxc[3]
        v3sigma3_uu_uu_uu = v3sigma3[:,0]
        v3sigma3_uu_uu_ud = v3sigma3[:,1]
        v3sigma3_uu_uu_dd = v3sigma3[:,2]
        v3sigma3_uu_ud_ud = v3sigma3[:,3]
        v3sigma3_uu_ud_dd = v3sigma3[:,4]
        v3sigma3_uu_dd_dd = v3sigma3[:,5]
        v3sigma3_ud_ud_ud = v3sigma3[:,6]
        v3sigma3_ud_ud_dd = v3sigma3[:,7]
        v3sigma3_ud_dd_dd = v3sigma3[:,8]
        v3sigma3_dd_dd_dd = v3sigma3[:,9]

        v2sigma2_1_uu_uu_u = np.vstack(
            [v3rhosigma2_u_uu_uu,
             (v3sigma3_uu_uu_uu * rho_u[1:4] * 2 +
              v3sigma3_uu_uu_ud * rho_d[1:4])]
        )
        v2sigma2_1_uu_uu_d = np.vstack(
            [v3rhosigma2_d_uu_uu,
             (v3sigma3_uu_uu_dd * rho_d[1:4] * 2 +
              v3sigma3_uu_uu_ud * rho_u[1:4])]
        )

        v2sigma2_1_uu_ud_u = np.vstack(
            [v3rhosigma2_u_uu_ud,
             (v3sigma3_uu_uu_ud * rho_u[1:4] * 2 +
              v3sigma3_uu_ud_ud * rho_d[1:4])]
        )
        v2sigma2_1_uu_ud_d = np.vstack(
            [v3rhosigma2_d_uu_ud,
             (v3sigma3_uu_ud_dd * rho_d[1:4] * 2 +
              v3sigma3_uu_ud_ud * rho_u[1:4])]
        )

        v2sigma2_1_uu_dd_u = np.vstack(
            [v3rhosigma2_u_uu_dd,
             (v3sigma3_uu_uu_dd * rho_u[1:4] * 2 +
              v3sigma3_uu_ud_dd * rho_d[1:4])]
        )
        v2sigma2_1_uu_dd_d = np.vstack(
            [v3rhosigma2_d_uu_dd,
             (v3sigma3_uu_dd_dd * rho_d[1:4] * 2 +
              v3sigma3_uu_ud_dd * rho_u[1:4])]
        )

        v2sigma2_1_ud_ud_u = np.vstack(
            [v3rhosigma2_u_ud_ud,
             (v3sigma3_uu_ud_ud * rho_u[1:4] * 2 +
              v3sigma3_ud_ud_ud * rho_d[1:4])]
        )
        v2sigma2_1_ud_ud_d = np.vstack(
            [v3rhosigma2_d_ud_ud,
             (v3sigma3_ud_ud_dd * rho_d[1:4] * 2 +
              v3sigma3_ud_ud_ud * rho_u[1:4])]
        )

        v2sigma2_1_ud_dd_u = np.vstack(
            [v3rhosigma2_u_ud_dd,
             (v3sigma3_uu_ud_dd * rho_u[1:4] * 2 +
              v3sigma3_ud_ud_dd * rho_d[1:4])]
        )
        v2sigma2_1_ud_dd_d = np.vstack(
            [v3rhosigma2_d_ud_dd,
             (v3sigma3_ud_dd_dd * rho_d[1:4] * 2 +
              v3sigma3_ud_ud_dd * rho_u[1:4])]
        )

        v2sigma2_1_dd_dd_u = np.vstack(
            [v3rhosigma2_u_dd_dd,
             (v3sigma3_uu_dd_dd * rho_u[1:4] * 2 +
              v3sigma3_ud_dd_dd * rho_d[1:4])]
        )
        v2sigma2_1_dd_dd_d = np.vstack(
            [v3rhosigma2_d_dd_dd,
             (v3sigma3_dd_dd_dd * rho_d[1:4] * 2 +
              v3sigma3_ud_dd_dd * rho_u[1:4])]
        )

        v2sigma2_1 = (
            v2sigma2_1_uu_uu_u,
            v2sigma2_1_uu_uu_d,
            v2sigma2_1_uu_ud_u,
            v2sigma2_1_uu_ud_d,
            v2sigma2_1_uu_dd_u,
            v2sigma2_1_uu_dd_d,
            v2sigma2_1_ud_ud_u,
            v2sigma2_1_ud_ud_d,
            v2sigma2_1_ud_dd_u,
            v2sigma2_1_ud_dd_d,
            v2sigma2_1_dd_dd_u,
            v2sigma2_1_dd_dd_d,
        )

        if xctype == 'MGGA':
            raise NotImplementedError
    else:
        raise KeyError
    return (v2rho2_1, v2rhosigma_1, v2sigma2_1) + (None,) * 7
