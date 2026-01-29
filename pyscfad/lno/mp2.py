# Copyright 2023-2026 The PySCFAD Authors
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

'''Impurity MP2 solver.
'''

from functools import reduce
import numpy
from pyscf.lib import logger
from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.cc import dfccsd
from pyscfad.lno import lno_base
from pyscfad.lno.ccsd import (
    _ChemistsERIs,
    mp2_fragment_energy,
    get_maskact,
)

class RCCSD(dfccsd.RCCSD):
    def ao2mo(self, mo_coeff=None, fockao=None):
        return _make_df_eris_incore(self, mo_coeff, fockao)

def _make_df_eris_incore(cc, mo_coeff=None, fockao=None):
    eris = _ChemistsERIs()
    eris._common_init_(cc, mo_coeff, fockao)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    with_df = cc.with_df
    naux = with_df.get_naoaux()

    mo = np.asarray(eris.mo_coeff)
    ijslice = (0, nocc, nocc, nmo)
    eri1 = with_df._cderi
    Lov = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1')

    eris.ovov = np.dot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
    return eris

def impurity_solve(mf, mo_coeff, lo_coeff, eris=None, frozen=None,
                   verbose_imp=0):
    log = logger.new_logger(mf)
    maskocc = mf.mo_occ > lno_base.THRESH_OCC
    nocc = numpy.count_nonzero(maskocc)
    nmo = mf.mo_occ.size

    frozen, maskact = get_maskact(frozen, nmo)

    orbfrzocc = mo_coeff[:,~maskact& maskocc]
    orbactocc = mo_coeff[:, maskact& maskocc]
    orbactvir = mo_coeff[:, maskact&~maskocc]
    orbfrzvir = mo_coeff[:,~maskact&~maskocc]
    nfrzocc, nactocc, nactvir, nfrzvir = [orb.shape[1]
                                          for orb in [orbfrzocc,orbactocc,
                                                      orbactvir,orbfrzvir]]
    nlo = lo_coeff.shape[1]
    s1e = eris.s1e
    prjlo = reduce(np.dot, (lo_coeff.T, s1e, orbactocc))

    log.info('    impsol:  %d LOs  %d/%d MOs  %d occ  %d vir',
             nlo, nactocc+nactvir, nmo, nactocc, nactvir)

    # solve impurity problem
    mcc = RCCSD(mf, mo_coeff=mo_coeff, frozen=frozen)
    mcc.e_hf = mf.e_tot  #avoid MP2 recompute e_hf
    imp_eris = mcc.ao2mo(fockao=eris.fock)

    # MP2 fragment energy
    t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
    elcorr_pt2 = mp2_fragment_energy(imp_eris, t2, prjlo)

    t1 = t2 = imp_eris = mcc = None
    del log
    return (elcorr_pt2,)

class LNOMP2(lno_base.LNO):
    def __init__(self, mf, thresh=1e-4, frozen=None, **kwargs):
        super().__init__(mf, thresh=thresh, frozen=frozen, **kwargs)
        self.efrag_pt2 = None

    def impurity_solve(self, mf, mo_coeff, lo_coeff, eris=None, frozen=None):
        return impurity_solve(mf, mo_coeff, lo_coeff, eris=eris, frozen=frozen,
                              verbose_imp=self.verbose_imp)

    def _post_proc(self, frag_res, frag_wghtlist):
        ''' Post processing results returned by `impurity_solve` collected in `frag_res`.
        '''
        efrag_pt2 = 0.0
        for i, res in enumerate(frag_res):
            if res is not None:
                efrag_pt2 += res[0] * frag_wghtlist[i]
        self.efrag_pt2  = efrag_pt2

    @property
    def e_corr(self):
        return self.efrag_pt2
