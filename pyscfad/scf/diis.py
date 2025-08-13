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

import numpy
from pyscf.scf import diis as pyscf_cdiis
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.ops import jit, vmap
from pyscfad import lib
from pyscfad.lib import logger

class CDIIS(lib.diis.DIIS, pyscf_cdiis.CDIIS):
    def __init__(self, mf=None, filename=None, Corth=None):
        pyscf_cdiis.CDIIS.__init__(self, mf=mf, filename=filename, Corth=Corth)
        self.incore = True

    def update(self, s, d, f, *args, **kwargs):
        errvec = get_err_vec(s, d, f, self.Corth)
        # no need to trace error vectors
        errvec = ops.stop_grad(errvec)
        logger.debug1(self, 'diis-norm(errvec)=%g', np.linalg.norm(errvec))
        f_prev = kwargs.get('f_prev', None)
        if abs(self.damp) < 1e-6 or f_prev is None:
            xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
        else:
            f = f*(1-self.damp) + f_prev*self.damp
            xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return xnew

@jit
def get_err_vec_orig(s, d, f):
    def _get_errvec(s, d, f):
        sdf = s @ d @ f
        return (sdf.conj().T - sdf).ravel()

    if f.ndim == 2:
        errvec = _get_errvec(s, d, f)

    elif f.ndim == 3 and s.ndim == 3:
        errvec = vmap(_get_errvec,
                      signature='(i,j),(i,j),(i,j)->(k)')(s, d, f)
        errvec = np.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = vmap(_get_errvec, in_axes=(None,0,0),
                      signature='(i,j),(i,j)->(k)')(s, d, f)
        errvec = np.hstack(errvec)

    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec

@jit
def get_err_vec_orth(s, d, f, Corth):
    def _get_errvec(s, d, f, c):
        sdf = c.conj().T @ s @ d @ f @ c
        return (sdf.conj().T - sdf).ravel()

    if f.ndim == 2:
        errvec = _get_errvec(s, d, f, Corth)

    elif f.ndim == 3 and s.ndim == 3:
        errvec = vmap(_get_errvec,
                      signature='(i,j),(i,j),(i,j),(i,j)->(k)')(s, d, f, Corth)
        errvec = np.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = vmap(_get_errvec, in_axes=(None,0,0,0),
                      signature='(i,j),(i,j),(i,j)->(k)')(s, d, f, Corth)
        errvec = np.hstack(errvec)

    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec

def get_err_vec(s, d, f, Corth=None):
    if Corth is None:
        return get_err_vec_orig(s, d, f)
    else:
        return get_err_vec_orth(s, d, f, Corth)

SCFDIIS = SCF_DIIS = DIIS = CDIIS
