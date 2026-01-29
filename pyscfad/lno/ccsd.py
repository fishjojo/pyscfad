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

'''Impurity solver for LNO CCSD/CCSD(T).
'''

import numpy
from functools import reduce

from pyscf.lib import logger
from pyscf.mp.mp2 import _mo_without_core

from pyscfad import numpy as np
from pyscfad import lib
from pyscfad.ao2mo import _ao2mo
from pyscfad.cc import dfccsd, dfdcsd
from pyscfad.lno import lno_base
from pyscfad.lno import ccsd_t as ccsd_t_mod

class RCCSD(dfccsd.RCCSD):
    def ao2mo(self, mo_coeff=None, fockao=None):
        return _make_df_eris_incore(self, mo_coeff, fockao)

class RDCSD(dfdcsd.RDCSD):
    def ao2mo(self, mo_coeff=None, fockao=None):
        return _make_df_eris_incore(self, mo_coeff, fockao)

class _ChemistsERIs(dfccsd._ChemistsERIs):
    def _common_init_(self, mycc, mo_coeff=None, fockao=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)

        if fockao is None:
            dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
            vhf = mycc._scf.get_veff(mycc.mol, dm)
            fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        #self.e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)
        self.e_hf = mycc._scf.e_tot
        self.nocc = mycc.nocc
        self.mol = mycc.mol

        self.mo_energy = np.diagonal(self.fock).real
        return self

def _make_df_eris_incore(cc, mo_coeff=None, fockao=None):
    eris = _ChemistsERIs()
    eris._common_init_(cc, mo_coeff, fockao)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    with_df = cc.with_df
    naux = with_df.get_naoaux()

    mo = np.asarray(eris.mo_coeff)
    ijslice = (0, nmo, 0, nmo)
    eri1 = with_df._cderi
    # pylint: disable=too-many-function-args
    Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1').reshape(-1,nmo,nmo)
    Loo = Lpq[:,:nocc,:nocc].reshape(naux,-1)
    Lov = Lpq[:,:nocc,nocc:].reshape(naux,-1)
    eris.Lvv = Lvv = lib.pack_tril(Lpq[:,nocc:,nocc:])

    eris.oooo = np.dot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo = np.dot(Lov.T, Loo).reshape(nocc,nvir,nocc,nocc)
    ovov = np.dot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
    eris.ovov = ovov
    eris.ovvo = ovov.transpose(0,1,3,2)

    oovv = np.dot(Loo.T, Lvv)
    eris.oovv = lib.unpack_tril(oovv).reshape(nocc,nocc,nvir,nvir)
    eris.ovvv = np.dot(Lov.T, Lvv).reshape(nocc,nvir,-1)
    Loo = Lov = Lpq = None
    return eris


def impurity_solve(mf, mo_coeff, lo_coeff, eris=None, frozen=None,
                   verbose_imp=0, ccsd_t=False, dcsd=False):
    r'''Solve impurity problem and calculate local correlation energy.

    Args:
        mo_coeff : array
            MOs for which the impurity problem is solved.
        lo_coeff : array
            LOs on the current fragment.
        ccsd_t : bool
            If set to ``True``, CCSD(T) energy is calculated and returned as the third
            item (0 is returned otherwise). Default is ``False``.
        dcsd : bool
            If set to ``True``, the DCSD correlation energy is computed instead
            of CCSD. Default is ``False``.
        frozen : int or list, optional
            Same syntax as ``frozen`` in MP2, CCSD, etc.
        verbose_imp : int
            Verbosity for impurity solver printing. Default is 0.

    Return:
        e_loc_corr_pt2, e_loc_corr_ccsd, e_loc_corr_ccsd_t:
            Local correlation energy at MP2, CCSD, and CCSD(T) levels. Note that
            the CCSD(T) energy is 0 unless ``ccsd_t`` is set to True.
    '''
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
    if dcsd:
        mcc = RDCSD(mf, mo_coeff=mo_coeff, frozen=frozen)
    else:
        mcc = RCCSD(mf, mo_coeff=mo_coeff, frozen=frozen)
    mcc.e_hf = mf.e_tot  #avoid MP2 recompute e_hf
    imp_eris = mcc.ao2mo(fockao=eris.fock)

    # MP2 fragment energy
    t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
    elcorr_pt2 = mp2_fragment_energy(imp_eris, t2, prjlo)

    # CCSD fragment energy
    t1, t2 = mcc.kernel(eris=imp_eris, t1=t1, t2=t2)[1:]
    elcorr_cc = ccsd_fragment_energy(imp_eris, t1, t2, prjlo)

    if ccsd_t and not dcsd:
        #for tests
        #from pyscfad.lno import ccsd_t_slow
        #elcorr_cc_t = ccsd_t_slow.kernel(mcc, imp_eris, prjlo, t1=t1, t2=t2, verbose=verbose_imp)
        #elcorr_cc_t = ccsd_t_slow.iterative_kernel(
        #    mcc, imp_eris, prjlo, t1=t1, t2=t2, verbose=verbose_imp)
        #from pyscfad.cc import gccsd_t
        #elcorr_cc_t = gccsd_t.kernel(mcc, prjlo, t1=t1, t2=t2)
        elcorr_cc_t = ccsd_t_mod.kernel(mcc, imp_eris, prjlo, t1=t1, t2=t2, verbose=verbose_imp)
    else:
        elcorr_cc_t = 0.

    t1 = t2 = imp_eris = mcc = None
    del log
    return (elcorr_pt2, elcorr_cc, elcorr_cc_t)

def get_maskact(frozen, nmo):
    if frozen is None:
        frozen = 0
    elif len(frozen) == 0:
        frozen = 0

    if numpy.isscalar(frozen):
        maskact = numpy.hstack([numpy.zeros(frozen,dtype=bool),
                                numpy.ones(nmo-frozen,dtype=bool)])
    else:
        maskact = numpy.array([i not in frozen for i in range(nmo)])
    return frozen, maskact

def mp2_fragment_energy(eris, t2, prj):
    m = np.dot(prj.T, prj)
    ovov = np.asarray(eris.ovov)
    eij  = 2*np.einsum('pjab,qajb->pq', t2, ovov)
    eij +=  -np.einsum('pjab,qbja->pq', t2, ovov)
    e2 = np.einsum('ij,ij', eij, m)
    return e2

def ccsd_fragment_energy(eris, t1, t2, prj):
    nocc = t1.shape[0]
    m = np.dot(prj.T, prj)
    fov = eris.fock[:nocc,nocc:]
    eij = 2*np.einsum('ia,ja->ij', t1, fov)
    tau = np.einsum('ia,jb->ijab', t1, t1)
    tau += t2
    ovov = np.asarray(eris.ovov)
    eij += 2*np.einsum('pjab,qajb->pq', tau, ovov)
    eij -= np.einsum('pjab,qbja->pq', tau, ovov)
    e2 = np.einsum('ij,ij', eij, m)
    return e2

class LNOCCSD(lno_base.LNO):
    def __init__(self, mf, thresh=1e-4, frozen=None, fock=None, s1e=None, **kwargs):
        super().__init__(mf, thresh=thresh, frozen=frozen, fock=fock, s1e=s1e, **kwargs)
        self.efrag_cc = None
        self.efrag_pt2 = None
        self.efrag_cc_t = None
        self.ccsd_t = False
        self.dcsd = False

    def impurity_solve(self, mf, mo_coeff, lo_coeff, eris=None, frozen=None):
        return impurity_solve(mf, mo_coeff, lo_coeff, eris=eris, frozen=frozen,
                              verbose_imp=self.verbose_imp, ccsd_t=self.ccsd_t,
                              dcsd=self.dcsd)

    def _post_proc(self, frag_res, frag_wghtlist):
        ''' Post processing results returned by ``impurity_solve`` collected in ``frag_res``.
        '''
        efrag_pt2 = efrag_cc = efrag_cc_t = 0.0
        for i, res in enumerate(frag_res):
            if res is not None:
                efrag_pt2 += res[0] * frag_wghtlist[i]
                efrag_cc += res[1] * frag_wghtlist[i]
                efrag_cc_t += res[2] * frag_wghtlist[i]
        self.efrag_pt2  = efrag_pt2
        self.efrag_cc   = efrag_cc
        self.efrag_cc_t = efrag_cc_t

    @property
    def e_corr(self):
        return self.e_corr_ccsd + self.e_corr_ccsd_t

    @property
    def e_corr_ccsd(self):
        e_corr = self.efrag_cc
        return e_corr

    @property
    def e_corr_pt2(self):
        e_corr = self.efrag_pt2
        return e_corr

    @property
    def e_corr_ccsd_t(self):
        e_corr = self.efrag_cc_t
        return e_corr

    @property
    def e_tot_ccsd(self):
        return self.e_corr_ccsd + self._scf.e_tot

    @property
    def e_tot_pt2(self):
        return self.e_corr_pt2 + self._scf.e_tot

    def e_corr_pt2corrected(self, ept2):
        return self.e_corr - self.e_corr_pt2 + ept2

    def e_tot_pt2corrected(self, ept2):
        return self._scf.e_tot + self.e_corr_pt2corrected(ept2)

    def e_corr_ccsd_pt2corrected(self, ept2):
        return self.e_corr_ccsd - self.e_corr_pt2 + ept2

    def e_tot_ccsd_pt2corrected(self, ept2):
        return self._scf.e_tot_ccsd + self.e_corr_pt2corrected(ept2)

    def e_corr_ccsd_t_pt2corrected(self, ept2):
        return self.e_corr_ccsd_t - self.e_corr_pt2 + ept2

    def e_tot_ccsd_t_pt2corrected(self, ept2):
        return self._scf.e_tot_ccsd_t + self.e_corr_pt2corrected(ept2)

class LNOCCSD_T(LNOCCSD):
    def __init__(self, mf, thresh=1e-4, frozen=None, **kwargs):
        super().__init__(mf, thresh=thresh, frozen=frozen, **kwargs)
        self.ccsd_t = True
