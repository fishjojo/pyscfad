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
SCF with k-point sampling
"""
from __future__ import annotations
import numpy

from pyscfad.typing import ArrayLike, Array
from pyscfad import numpy as np
from pyscfad.ops import vmap
from pyscfad.lib import logger
from pyscfad.scf.hf_lite import SCFLite
from pyscfad.dft.rks import VXC
from pyscfad.pbc.gto import CellLite as Cell

def get_occ(
    mf: KSCF,
    mo_energy_kpts: ArrayLike | None = None,
    mo_coeff_kpts: ArrayLike | None = None,
) -> Array:
    if mf.sigma is not None and mf.sigma > 0:
        raise NotImplementedError

    if mo_energy_kpts is None:
        mo_energy_kpts = mf.mo_energy

    nocc = mf.tot_electrons // 2

    mo_energy = np.sort(np.hstack(mo_energy_kpts))
    fermi = mo_energy[nocc-1]
    mo_occ_kpts = np.where(mo_energy_kpts <= fermi, 2., 0.)

    if mf.verbose >= logger.DEBUG:
        if nocc < mo_energy.size:
            logger.debug(mf, "HOMO = %.12g  LUMO = %.12g",
                        mo_energy[nocc-1], mo_energy[nocc])
        else:
            logger.debug(mf, "HOMO = %.12g", mo_energy[nocc-1])

        numpy.set_printoptions(threshold=len(mo_energy))
        logger.debug(mf, "     k-point                  mo_energy")
        for k, kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, "  %2d (%6.3f %6.3f %6.3f)   %s",
                         k, kpt[0], kpt[1], kpt[2], mo_energy_kpts[k])
        numpy.set_printoptions(threshold=1000)

    return mo_occ_kpts


class KSCF(SCFLite):
    """Base class for SCF methods with k-point sampling.
    """
    def __init__(
        self,
        cell: Cell,
        kpts: ArrayLike | None = None,
        **kwargs,
    ):
        if kpts is None:
            kpts = np.zeros((1,3))
        self.kpts = np.asarray(kpts, dtype=float).reshape(-1,3)

        super().__init__(cell, **kwargs)

    @property
    def cell(self) -> Cell:
        return self.mol

    def get_ovlp(
        self,
        cell: Cell | None = None,
        kpts: ArrayLike | None = None,
    ) -> Array:
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts = self.kpts
        return cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts)

    def make_rdm1(
        self,
        mo_coeff: ArrayLike | None = None,
        mo_occ: ArrayLike | None = None,
        **kwargs,
    ) -> Array:
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        return vmap(super().make_rdm1)(mo_coeff, mo_occ)

    def _eigh(self, h: ArrayLike, s: ArrayLike) -> tuple[Array, Array]:
        return vmap(super()._eigh)(h, s)

    def get_grad(
        self,
        mo_coeff: ArrayLike,
        mo_occ: ArrayLike,
        fock: ArrayLike | None = None,
    ) -> Array:
        if fock is None:
            dm = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_fock(dm=dm)
        return vmap(super().get_grad)(mo_coeff, mo_occ, fock)

    def energy_elec(
        self,
        dm: ArrayLike | None = None,
        h1e: ArrayLike | None = None,
        vhf: ArrayLike | VXC | None = None,
    ) -> tuple[float, float]:
        if dm is None:
            dm = self.make_rdm1()
        if h1e is None:
            h1e = self.get_hcore(self.cell)
        if vhf is None or (self.veff_with_ecoul and getattr(vhf, "ecoul", None) is None):
            vhf = self.get_veff(self.cell, dm)

        weight = 1. / len(self.kpts)
        e1 = weight * np.einsum("kij,kji", h1e, dm).real
        if hasattr(vhf, "ecoul"):
            ecoul = vhf.ecoul.real
        else:
            ecoul = weight * np.einsum("kij,kji", vhf, dm).real * 0.5

        self.scf_summary["e1"] = e1
        self.scf_summary["e2"] = ecoul
        logger.debug(self, "E1 = %s  E_coul = %s", e1, ecoul)
        return e1+ecoul, ecoul

    get_occ = get_occ

KSCFLite = KSCF
