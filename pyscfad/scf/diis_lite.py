# Copyright 2021-2026 The PySCFAD Authors
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
Jittable CDIIS
"""
from pyscfad import numpy as np
from pyscfad.ops import vmap
from pyscfad.lib.diis_lite import DIISLite

class CDIIS(DIISLite):
    def __init__(self, f, space=8, Corth=None):
        super().__init__(f.ravel(), space=space)
        self.Corth = Corth
        self.damp = 0

    def update(self, s, d, f, *args, **kwargs):
        errvec = get_err_vec(s, d, f, self.Corth)
        f_prev = kwargs.get("f_prev", None)
        if abs(self.damp) < 1e-6 or f_prev is None:
            xnew = DIISLite.update(self, f.ravel(), xerr=errvec)
        else:
            f = f*(1-self.damp) + f_prev*self.damp
            xnew = DIISLite.update(self, f.ravel(), xerr=errvec)
        return xnew.reshape(f.shape)

def get_err_vec_orig(s, d, f):
    def _get_errvec(s, d, f):
        sdf = s @ d @ f
        return (sdf.conj().T - sdf).ravel()

    if f.ndim == 2:
        errvec = _get_errvec(s, d, f)

    elif f.ndim == 3 and s.ndim == 3:
        errvec = vmap(_get_errvec,
                      signature="(i,j),(i,j),(i,j)->(k)")(s, d, f)
        errvec = np.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = vmap(_get_errvec, in_axes=(None,0,0),
                      signature="(i,j),(i,j)->(k)")(s, d, f)
        errvec = np.hstack(errvec)

    else:
        raise RuntimeError("Unknown SCF DIIS type")
    return errvec

def get_err_vec_orth(s, d, f, Corth):
    def _get_errvec(s, d, f, c):
        sdf = c.conj().T @ s @ d @ f @ c
        return (sdf.conj().T - sdf).ravel()

    if f.ndim == 2:
        errvec = _get_errvec(s, d, f, Corth)

    elif f.ndim == 3 and s.ndim == 3:
        errvec = vmap(_get_errvec,
                      signature="(i,j),(i,j),(i,j),(i,j)->(k)")(s, d, f, Corth)
        errvec = np.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = vmap(_get_errvec, in_axes=(None,0,0,0),
                      signature="(i,j),(i,j),(i,j)->(k)")(s, d, f, Corth)
        errvec = np.hstack(errvec)

    else:
        raise RuntimeError("Unknown SCF DIIS type")
    return errvec

def get_err_vec(s, d, f, Corth=None):
    if Corth is None:
        return get_err_vec_orig(s, d, f)
    else:
        return get_err_vec_orth(s, d, f, Corth)

SCFDIIS = SCF_DIIS = DIIS = CDIIS
