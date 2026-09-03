# Copyright 2026 The PySCFAD Authors
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
UMP2 with spatial integrals
"""
from pyscf import __config__ as pyscf_config
from pyscf.mp import ump2 as pyscf_ump2
from pyscfad import numpy as np
from pyscfad import pytree
from pyscfad import ao2mo
from pyscfad.lib import logger
from pyscfad.mp import mp2

WITH_T2 = getattr(pyscf_config, 'mp_ump2_with_t2', True)

def _t2_amplitudes(g, eia_i, eia_j):
    """Non-antisymmetrized amplitudes ``t[i,j,a,b] = (ia|jb)^*/D[i,j,a,b]``
    from the ``(ia|jb)`` integral block ``g``.
    """
    g = g.transpose(0,2,1,3)
    d = eia_i[:,None,:,None] + eia_j[None,:,None,:]
    return g.conj() / d, g

def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)

    if mo_energy is None:
        mo_energy = eris.mo_energy

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb
    mo_ea, mo_eb = mo_energy
    eia_a = mo_ea[:nocca,None] - mo_ea[None,nocca:]
    eia_b = mo_eb[:noccb,None] - mo_eb[None,noccb:]

    eris_ovov = np.asarray(eris.ovov).reshape(nocca,nvira,nocca,nvira)
    eris_ovOV = np.asarray(eris.ovOV).reshape(nocca,nvira,noccb,nvirb)
    eris_OVOV = np.asarray(eris.OVOV).reshape(noccb,nvirb,noccb,nvirb)

    t2aa, gaa = _t2_amplitudes(eris_ovov, eia_a, eia_a)
    t2ab, gab = _t2_amplitudes(eris_ovOV, eia_a, eia_b)
    t2bb, gbb = _t2_amplitudes(eris_OVOV, eia_b, eia_b)
    # same-spin amplitudes are antisymmetric with respect to the virtual indices
    t2aa = t2aa - t2aa.transpose(0,1,3,2)
    t2bb = t2bb - t2bb.transpose(0,1,3,2)

    emp2_ss  = np.einsum('ijab,ijab', t2aa, gaa) * .5
    emp2_ss += np.einsum('ijab,ijab', t2bb, gbb) * .5
    emp2_os  = np.einsum('iJaB,iJaB', t2ab, gab)

    # PySCF tags the same/opposite spin components onto the energy with
    # lib.tag_array, which would break the AD. They are stored on ``mp``
    # instead, as done by MP2Base.kernel.
    mp.e_corr_ss = emp2_ss = emp2_ss.real
    mp.e_corr_os = emp2_os = emp2_os.real

    if with_t2:
        t2 = (t2aa, t2ab, t2bb)
    else:
        t2 = None
    return emp2_ss + emp2_os, t2

def energy(mp, t2, eris):
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    eris_ovov = np.asarray(eris.ovov).reshape(nocca,nvira,nocca,nvira)
    eris_OVOV = np.asarray(eris.OVOV).reshape(noccb,nvirb,noccb,nvirb)
    eris_ovOV = np.asarray(eris.ovOV).reshape(nocca,nvira,noccb,nvirb)
    ess  = 0.25 * np.einsum('ijab,iajb->', t2aa, eris_ovov)
    ess -= 0.25 * np.einsum('ijab,ibja->', t2aa, eris_ovov)
    ess += 0.25 * np.einsum('ijab,iajb->', t2bb, eris_OVOV)
    ess -= 0.25 * np.einsum('ijab,ibja->', t2bb, eris_OVOV)
    eos  =        np.einsum('iJaB,iaJB->', t2ab, eris_ovOV)
    return (ess + eos).real

def update_amps(mp, t2, eris):
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:] + mp.level_shift
    mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:] + mp.level_shift

    focka, fockb = eris.fock
    fooa = focka[:nocca,:nocca] - np.diag(mo_ea_o)
    foob = fockb[:noccb,:noccb] - np.diag(mo_eb_o)
    fvva = focka[nocca:,nocca:] - np.diag(mo_ea_v)
    fvvb = fockb[noccb:,noccb:] - np.diag(mo_eb_v)

    u2aa  = np.einsum('ijae,be->ijab', t2aa, fvva)
    u2bb  = np.einsum('ijae,be->ijab', t2bb, fvvb)
    u2ab  = np.einsum('iJaE,BE->iJaB', t2ab, fvvb)
    u2ab += np.einsum('iJeA,be->iJbA', t2ab, fvva)
    u2aa -= np.einsum('imab,mj->ijab', t2aa, fooa)
    u2bb -= np.einsum('imab,mj->ijab', t2bb, foob)
    u2ab -= np.einsum('iMaB,MJ->iJaB', t2ab, foob)
    u2ab -= np.einsum('mIaB,mj->jIaB', t2ab, fooa)

    eris_ovov = np.asarray(eris.ovov).reshape(nocca,nvira,nocca,nvira).conj() * .5
    eris_OVOV = np.asarray(eris.OVOV).reshape(noccb,nvirb,noccb,nvirb).conj() * .5
    eris_ovOV = np.asarray(eris.ovOV).reshape(nocca,nvira,noccb,nvirb).conj()
    u2aa += eris_ovov.transpose(0,2,1,3) - eris_ovov.transpose(0,2,3,1)
    u2bb += eris_OVOV.transpose(0,2,1,3) - eris_OVOV.transpose(0,2,3,1)
    u2ab += eris_ovOV.transpose(0,2,1,3)
    u2aa = u2aa + u2aa.transpose(1,0,3,2)
    u2bb = u2bb + u2bb.transpose(1,0,3,2)

    eia_a = mo_ea_o[:,None] - mo_ea_v[None,:]
    eia_b = mo_eb_o[:,None] - mo_eb_v[None,:]
    u2aa /= eia_a[:,None,:,None] + eia_a[None,:,None,:]
    u2ab /= eia_a[:,None,:,None] + eia_b[None,:,None,:]
    u2bb /= eia_b[:,None,:,None] + eia_b[None,:,None,:]
    return u2aa, u2ab, u2bb

class UMP2(pytree.PytreeNode, pyscf_ump2.UMP2):
    _dynamic_attr = {'_scf', 'mol'}

    def ao2mo(self, mo_coeff=None):
        eris = _ChemistsERIs()
        eris._common_init_(self, mo_coeff)
        moa, mob = eris.mo_coeff

        nocca, noccb = self.nocc
        occa = np.asarray(moa[:,:nocca])
        vira = np.asarray(moa[:,nocca:])
        occb = np.asarray(mob[:,:noccb])
        virb = np.asarray(mob[:,noccb:])

        eri_ao = self._scf._eri
        eris.ovov = ao2mo.general(eri_ao, (occa,vira,occa,vira))
        eris.ovOV = ao2mo.general(eri_ao, (occa,vira,occb,virb))
        eris.OVOV = ao2mo.general(eri_ao, (occb,virb,occb,virb))
        return eris

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        if self.verbose >= logger.WARN:
            self.check_sanity()

        self.dump_flags()

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        if eris is None:
            eris = self.ao2mo(mo_coeff)

        if self._scf.converged:
            # init_amps also sets e_corr_ss and e_corr_os
            self.e_corr, self.t2 = self.init_amps(mo_energy, mo_coeff, eris, with_t2)
        else:
            self.converged, self.e_corr, self.t2 = self._iterative_kernel(eris)
            # TODO SCS-MP2 for the non-canonical case
            self.e_corr_ss = 0
            self.e_corr_os = 0

        self._finalize()
        return self.e_corr, self.t2

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

    def nuc_grad_method(self):
        raise NotImplementedError

    def density_fit(self, auxbasis=None, with_df=None):
        raise NotImplementedError

    energy = energy
    update_amps = update_amps
    _iterative_kernel = mp2._iterative_kernel

MP2 = UMP2

class _ChemistsERIs(pyscf_ump2._ChemistsERIs):
    def _common_init_(self, mp, mo_coeff=None):
        self.mol = mp.mol
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
                               'You may need to call mf.kernel() to generate them.')

        mo_idx = mp.get_frozen_mask()
        mo_a = mo_coeff[0][:,mo_idx[0]]
        mo_b = mo_coeff[1][:,mo_idx[1]]
        self.mo_coeff = (mo_a, mo_b)

        if mo_coeff is mp._scf.mo_coeff and mp._scf.converged:
            self.mo_energy = (mp._scf.mo_energy[0][mo_idx[0]],
                              mp._scf.mo_energy[1][mo_idx[1]])
            self.fock = (np.diag(self.mo_energy[0]),
                         np.diag(self.mo_energy[1]))
        else:
            dm = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)
            vhf = mp._scf.get_veff(mp.mol, dm)
            fockao = mp._scf.get_fock(vhf=vhf, dm=dm)
            focka = mo_a.conj().T.dot(fockao[0]).dot(mo_a)
            fockb = mo_b.conj().T.dot(fockao[1]).dot(mo_b)
            self.fock = (focka, fockb)
            self.nocc = mp.nocc
            self.mo_energy = (focka.diagonal().real, fockb.diagonal().real)
        return self
