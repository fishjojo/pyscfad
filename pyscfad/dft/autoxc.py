# Copyright 2021-2025 The PySCFAD Authors
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

"""
Interface to autoxc
"""

from functools import cached_property, partial
import numpy
import autoxc
from autoxc import api
from pyscf.dft.libxc import (
    xc_type as libxc_xc_type,
    is_hybrid_xc as libxc_is_hybrid_xc,
    _libxc_to_xcfun_indices,
)
from pyscfad import numpy as np
from pyscfad.ops import jit

__version__ = autoxc.__version__
__reference__ = "The autoxc authors, unpublished"

class XCFunctionalCache:
    def __init__(self, xc_code, spin=0, omega=None):
        assert isinstance(xc_code, dict)

        self.nfunc = len(xc_code)

        xc_arr = []
        facs = []
        params = []
        for k, v in xc_code.items():
            _xc = api.get_xc(k)
            xc_arr.append(_xc)
            facs.append(v.get("coeff"))
            params.append(v.get("params", _xc.params))

        self.xc_arr = xc_arr
        self.facs = facs
        self.params = params

    @cached_property
    def xc_type(self):
        if self.nfunc == 0:
            return "HF"

        has_lda  = numpy.asarray(list(map(api.is_lda,  self.xc_arr)), dtype=bool)
        has_gga  = numpy.asarray(list(map(api.is_gga,  self.xc_arr)), dtype=bool)
        has_mgga = numpy.asarray(list(map(api.is_mgga, self.xc_arr)), dtype=bool)

        if has_mgga.any():
            return "MGGA"
        elif has_gga.any():
            return "GGA"
        elif has_lda.any():
            return "LDA"
        else:
            return "UNKNOWN"

def _get_xc(xc_code, spin=0, omega=None):
    return XCFunctionalCache(xc_code, spin, omega)

def xc_type(xc_code):
    if isinstance(xc_code, str):
        return libxc_xc_type(xc_code)
    elif isinstance(xc_code, dict):
        xc = _get_xc(xc_code)
        return xc.xc_type
    else:
        raise KeyError(f"xc functional {xc_code} is not supported")

def gen_eval_xc(xc_code, deriv=0):
    xc = _get_xc(xc_code)
    fns = [api.gen_eval_xc(x, deriv=deriv) for x in xc.xc_arr]
    return fns, xc.facs, xc.params

def is_hybrid_xc(xc_code):
    if xc_code is None or isinstance(xc_code, dict):
        return False
    else:
        return libxc_is_hybrid_xc(xc_code)

@partial(jit, static_argnames=("spin", "deriv"))
def _eval_xc(xc_code, rho, spin=0, deriv=1, omega=None):
    if omega is not None:
        raise NotImplementedError
    xctype = xc_type(xc_code)

    if spin == 0:
        rhoa = rhob = rho * .5
    elif spin == 1:
        rhoa = rho[0]
        rhob = rho[1]
    else:
        raise KeyError(f"spin = {spin} is not supported")

    if xctype == "LDA":
        rho_autoxc = [rhoa, rhob]
    elif xctype == "GGA":
        sigma_aa = np.einsum("xg,xg->g", rhoa[1:4], rhoa[1:4])
        sigma_ab = np.einsum("xg,xg->g", rhoa[1:4], rhob[1:4])
        sigma_bb = np.einsum("xg,xg->g", rhob[1:4], rhob[1:4])
        rho_autoxc = [rhoa[0], rhob[0], sigma_aa, sigma_ab, sigma_bb]
    else:
        raise NotImplementedError

    out = []
    for i in range(deriv+1):
        exc = 0
        for fn, fac, param in zip(*gen_eval_xc(xc_code, deriv=i)):
            exc = exc + fn(rho_autoxc, param) * fac
        if i == 0:
            exc /= (rho_autoxc[0] + rho_autoxc[1])
        out.append(exc)
    out = np.vstack(out)
    if spin == 0:
        out = _eval_xc_u2r(out, xctype, deriv)
    return out

def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    outbuf = _eval_xc(xc_code, rho, spin=spin, deriv=deriv, omega=omega)
    exc = outbuf[0]
    vxc = fxc = kxc = None
    xctype = xc_type(xc_code)
    if xctype == "LDA" and spin == 0:
        if deriv > 0:
            vxc = [outbuf[1]]
        if deriv > 1:
            fxc = [outbuf[2]]
        if deriv > 2:
            kxc = [outbuf[3]]
    elif xctype == "GGA" and spin == 0:
        if deriv > 0:
            vxc = [outbuf[1], outbuf[2]]
        if deriv > 1:
            fxc = [outbuf[3], outbuf[4], outbuf[5]]
        if deriv > 2:
            kxc = [outbuf[6], outbuf[7], outbuf[8], outbuf[9]]
    elif xctype == "LDA" and spin == 1:
        if deriv > 0:
            vxc = [outbuf[1:3].T]
        if deriv > 1:
            fxc = [outbuf[3:6].T]
        if deriv > 2:
            kxc = [outbuf[6:10].T]
    elif xctype == "GGA" and spin == 1:
        if deriv > 0:
            vxc = [outbuf[1:3].T, outbuf[3:6].T]
        if deriv > 1:
            fxc = [outbuf[6:9].T, outbuf[9:15].T, outbuf[15:21].T]
        if deriv > 2:
            kxc = [outbuf[21:25].T, outbuf[25:34].T, outbuf[34:46].T, outbuf[46:56].T]
    else:
        raise NotImplementedError
    return exc, vxc, fxc, kxc

def eval_xc1(xc_code, rho, spin=0, deriv=1, omega=None):
    out = _eval_xc(xc_code, rho, spin=spin, deriv=deriv, omega=omega)
    idx = _libxc_to_xcfun_indices(xctype, spin, deriv)
    return out[idx]

def _eval_xc_u2r(exc, xctype, deriv):
    if deriv == 0:
        return exc

    out = [exc[0],]
    if xctype == "LDA":
        if deriv >= 1:
            vxc = (exc[1] + exc[2]) * .5
            out.append(vxc)
        elif deriv >= 2:
            fxc = (exc[3] + exc[4] * 2 + exc[5]) * .25
            out.append(fxc)
        elif deriv >= 3:
            raise NotImplementedError
            #kxc = exc[6] + exc[7] * 3 + exc[8] * 3 + exc[9]
            #out.append(kxc)
        elif deriv >= 4:
            raise NotImplementedError
            #lxc = exc[10] + exc[11] * 4 + exc[12] * 6 + exc[13] * 4 + exc[14]
            #out.append(lxc)

    elif xctype == "GGA":
        if deriv >= 1:
            vrho = (exc[1] + exc[2]) * .5
            vsigma = (exc[3] + exc[4] + exc[5]) * .25
            out = out + [vrho, vsigma]
        elif deriv >= 2:
            v2rho2 = (exc[6] + exc[7] * 2 + exc[8]) * .25
            v2rhosigma = (exc[9] + exc[10] + exc[11] + exc[12] + exc[13] + exc[14]) * .125
            v2sigma2 = (exc[15] + exc[16] * 2 + exc[17] * 2 + exc[18] + exc[19] * 2 + exc[20]) * .0625
            out = out + [v2rho2, v2rhosigma, v2sigma2]
        elif deriv >= 3:
            raise NotImplementedError
        elif deriv >= 4:
            raise NotImplementedError
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    return np.vstack(out)
