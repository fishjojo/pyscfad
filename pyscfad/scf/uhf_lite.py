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
Lightweight :mod:`~pyscfad.scf.uhf` module.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from pyscf.lib import module_method

from pyscfad import numpy as np
from pyscfad.ops import vmap
from pyscfad.scf import hf_lite, sp2, uhf

if TYPE_CHECKING:
    from pyscfad.typing import ArrayLike, Array
    from pyscfad.gto import MoleLite

def get_occ(
    mf: UHF,
    mo_energy: ArrayLike | None = None,
    mo_coeff: ArrayLike | None = None,
) -> Array:
    """Get the alpha and beta MO occupations.
    """
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    mask = mf.mo_mask(mo_energy, mo_coeff)
    return np.stack([hf_lite.fill_occ(mf, e, m, n)
                     for e, m, n in zip(mo_energy, mask, mf.nelec)])

class UHF(hf_lite.SCF):
    """Unrestricted Hartree-Fock.

    The density matrix, Fock matrix and MO quantities carry a leading
    spin axis for the alpha and beta electrons.
    """
    @property
    def nelec(self) -> tuple[int | Array, int | Array]:
        """Numbers of alpha and beta electrons."""
        nelectron = self.tot_electrons
        nalpha = (nelectron + self.mol.spin) // 2
        return nalpha, nelectron - nalpha

    def get_init_guess(
        self,
        mol: MoleLite | None = None,
        key: str = "minao",
        **kwargs,
    ) -> Array:
        dm = super().get_init_guess(mol=mol, key=key, **kwargs)
        if dm.ndim == 2:
            dm = np.stack((dm * .5, dm * .5))
        return dm

    def eig(self, h: ArrayLike, s: ArrayLike) -> tuple[Array, Array]:
        if h.ndim == 2: # e.g., the core Hamiltonian guess
            h = (h, h)
        return uhf.UHF.eig(self, h, s)

    def get_grad(
        self,
        mo_coeff: ArrayLike,
        mo_occ: ArrayLike,
        fock: ArrayLike | None = None,
    ) -> Array:
        if fock is None:
            fock = self.get_fock(dm=self.make_rdm1(mo_coeff, mo_occ))
        return vmap(hf_lite.get_grad)(mo_coeff, mo_occ, fock).ravel() * .5

    def make_rdm1_sp2(self, fock: Array, x: Array) -> Array:
        return vmap(sp2.make_rdm1)(fock, x, np.asarray(self.nelec))

    get_occ = get_occ
    get_veff = uhf.UHF.get_veff
    energy_elec = uhf.energy_elec
    make_rdm1 = module_method(uhf.make_rdm1, absences=["mo_coeff", "mo_occ"])

UHFLite = UHF
