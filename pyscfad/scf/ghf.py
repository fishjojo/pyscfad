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
Generalized Hartree-Fock
"""

import jax
from pyscf.scf import ghf as pyscf_ghf
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.lib import logger
from pyscfad.scf import hf

def get_jk(mol, dm, hermi=0,
           with_j=True, with_k=True, jkbuild=None, omega=None):
    assert callable(jkbuild), "jkbuild must be a callable function"

    dm = np.asarray(dm)
    nso = dm.shape[-1]
    nao = nso // 2
    dms = dm.reshape(-1,nso,nso)
    n_dm = dms.shape[0]

    dmaa = dms[:,:nao,:nao]
    dmab = dms[:,:nao,nao:]
    dmbb = dms[:,nao:,nao:]
    if with_k:
        if hermi:
            dms = np.stack((dmaa, dmbb, dmab))
        else:
            dmba = dms[:,nao:,:nao]
            dms = np.stack((dmaa, dmbb, dmab, dmba))
        # Note the off-diagonal block breaks the hermitian
        _hermi = 0
    else:
        dms = np.stack((dmaa, dmbb))
        _hermi = 1

    j1, k1 = jkbuild(mol, dms, _hermi, with_j, with_k, omega)

    vj = vk = None
    if with_j:
        vj = np.zeros((n_dm,nso,nso), dm.dtype)
        j = j1[0] + j1[1]
        vj = vj.at[:,:nao,:nao].set(j)
        vj = vj.at[:,nao:,nao:].set(j)
        vj = vj.reshape(dm.shape)

    if with_k:
        vk = np.zeros((n_dm,nso,nso), dm.dtype)
        vk = vk.at[:,:nao,:nao].set(k1[0])
        vk = vk.at[:,nao:,nao:].set(k1[1])
        vk = vk.at[:,:nao,nao:].set(k1[2])
        if hermi:
            vk = vk.at[:,nao:,:nao].set(k1[2].conj().transpose(0,2,1))
        else:
            vk = vk.at[:,nao:,:nao].set(k1[3])
        vk = vk.reshape(dm.shape)

    return vj, vk

def dip_moment(mol, dm, unit="Debye", origin=None, verbose=logger.NOTE):
    nao = mol.nao_nr()
    dma = dm[:nao,:nao]
    dmb = dm[nao:,nao:]
    return hf.dip_moment(mol, dma+dmb, unit=unit, verbose=verbose, origin=origin)

#TODO SOC
class GHF(hf.SCF):
    def get_hcore(self, mol=None, **kwargs):
        hcore = super().get_hcore(mol)
        hcore = jax.scipy.linalg.block_diag(hcore, hcore)
        return hcore

    def get_ovlp(self, mol=None):
        s = super().get_ovlp(mol)
        s = jax.scipy.linalg.block_diag(s, s)
        return s

    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        nao = mol.nao
        dm = np.asarray(dm)
        # nao = 0 for HF with custom Hamiltonian
        if dm.shape[-1] != nao * 2 and nao != 0:
            raise ValueError("Dimension inconsistent "
                             f"dm.shape = {dm.shape}, mol.nao = {nao}")

        def jkbuild(mol, dm, hermi, with_j, with_k, omega):
            return hf.SCF.get_jk(self, mol=mol, dm=dm, hermi=hermi,
                                 with_j=with_j, with_k=with_k, omega=omega)

        vj, vk = get_jk(mol, dm, hermi, with_j, with_k, jkbuild, omega)
        return vj, vk

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1, **kwargs):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj - vk
        else:
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj - vk + np.asarray(vhf_last)
        return vhf

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = self.mo_energy
        mo_energy = ops.to_numpy(mo_energy)
        if mo_coeff is not None:
            mo_coeff = ops.to_numpy(mo_coeff)
        return pyscf_ghf.get_occ(self, mo_energy, mo_coeff)

    def spin_square(self, mo_coeff=None, s=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff[:,self.mo_occ>0]
        if s is None:
            s = self.get_ovlp()
        mo_coeff = ops.to_numpy(mo_coeff)
        s = ops.to_numpy(s)
        return pyscf_ghf.spin_square(mo_coeff, s)

    def dip_moment(self, mol=None, dm=None, unit="Debye",
                   origin=None, verbose=logger.NOTE):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        return dip_moment(mol, dm, unit=unit, origin=origin, verbose=verbose)

    get_grad = pyscf_ghf.GHF.get_grad
    init_guess_by_minao = pyscf_ghf.GHF.init_guess_by_minao
    init_guess_by_atom = pyscf_ghf.GHF.init_guess_by_atom
    init_guess_by_huckel = pyscf_ghf.GHF.init_guess_by_huckel
    init_guess_by_mod_huckel = pyscf_ghf.GHF.init_guess_by_mod_huckel
    init_guess_by_sap = pyscf_ghf.GHF.init_guess_by_sap
    init_guess_by_chkfile = pyscf_ghf.GHF.init_guess_by_chkfile
    _finalize = pyscf_ghf.GHF._finalize
