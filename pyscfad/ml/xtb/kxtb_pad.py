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
k-point XTB with padding (for batched periodic calculations).

Follows the same strategy as :mod:`pyscfad.ml.xtb.xtb_pad` /
:mod:`pyscfad.ml.scf.hf_pad`: cells are padded with fake atoms
(``numbers == 0``) and a fixed-shape lattice-image grid, so heterogeneous
solids batch under ``jax.vmap`` with static shapes.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.lib import hermi_triu
from pyscfad.gto.mole import inter_distance

from pyscfad.xtb import xtb, kxtb, util
from pyscfad.xtb.data.elements import N_VALENCE_ARRAY
from pyscfad.xtb.data.radii import ATOMIC as ATOMIC_RADII
from pyscfad.ml.pbc.scf.khf_pad import KSCFPad

if TYPE_CHECKING:
    from typing import Any
    from pyscfad.typing import ArrayLike, Array
    from pyscfad.ml.pbc.gto import CellPad


def tot_valence_electrons(cell, charge: int | None = None, nkpts: int = 1):
    if charge is None:
        charge = cell.charge
    nelecs = N_VALENCE_ARRAY[cell.numbers]
    return np.sum(nelecs) * nkpts - charge


def EHT_PI_GFN1(
    cell,
    param: Any,
    atomic_radii: ArrayLike = ATOMIC_RADII,
    Ls: ArrayLike = np.zeros((1, 3)),
):
    """Padding-safe periodic EHT distance polynomial.

    Identical to :func:`pyscfad.xtb.kxtb.EHT_PI_GFN1` except that the
    covalent-radius sum is guarded: padding atoms (Z=0) have zero radius,
    which would otherwise produce 0/0 = NaN whose gradient survives masking.
    """
    shpoly = param.shpoly

    rr = inter_distance(cell, Ls=Ls)

    z = cell.atom_charges()
    cov_radii = atomic_radii[z]
    RAB = cov_radii[:, None] + cov_radii[None, :]
    RAB = np.where(RAB > 1e-12, RAB, 1.0)

    rr = np.safe_sqrt(rr / RAB[None, :, :], thresh=1e-6)

    i, j = util.atom_to_bas_indices_2d(cell)
    RR = rr[:, i, j]
    PI = (1 + shpoly[None, :, None] * RR) * (1 + shpoly[None, None, :] * RR)
    return PI


class KXTB(kxtb.KXTB, KSCFPad):
    @property
    def tot_electrons(self):
        return tot_valence_electrons(self.cell, nkpts=len(self.kpts))


class GFN1KXTB(kxtb.GFN1KXTB, KXTB):
    """Padded GFN1-XTB with k-point sampling."""

    def _get_gamma(self) -> Array:
        gamma = kxtb.GFN1KXTB._get_gamma(self)
        shl_mask = self.cell.shl_mask
        return np.where(np.outer(shl_mask, shl_mask), gamma, 0.0)

    def _energy_nuc(self, cell: CellPad | None = None) -> float:
        if cell is None:
            cell = self.cell
        param = self.param
        kf = param.kf
        zeff = param.zeff
        arep = param.arep

        Ls = cell.Ls
        r, r_inv = util.r_and_inv_r(cell, Ls=Ls)
        r_safe = np.where(r > 1e-6, r, 1.0)

        # padding-safe analogue of util.rcut_enuc_GFN1: padded entries have
        # zeff = arep = 0, which would give inf/NaN cutoffs
        zeff_max = np.max(zeff)
        zeff2 = np.maximum(zeff_max**2, 1e-30)
        arep_min = np.min(np.where(arep > 0, arep, np.inf))
        arep_min = np.where(np.isfinite(arep_min), arep_min, 1.0)
        rc = 20.0
        for _ in range(2):
            rc = (-np.log(rc * cell.precision / zeff2) / arep_min) ** (1.0 / kf)
        rcut = ops.stop_grad(np.where(zeff_max > 0, rc, 0.0))

        z_ab = zeff[:, None] * zeff[None, :]
        arep_ab = np.safe_sqrt(arep[:, None] * arep[None, :], thresh=1e-14)

        damp = np.where(r > 1e-6, np.exp(-arep_ab[None, ...] * r_safe**kf), 0.0)
        enuc_ab = np.where(r < rcut, z_ab[None, ...] * damp * r_inv, 0.0)
        enuc = 0.5 * np.sum(enuc_ab)
        return enuc

    def get_init_guess(
        self,
        cell: CellPad | None = None,
        key: str = "refocc",
        s1e: ArrayLike | None = None,
    ) -> Array:
        if cell is None:
            cell = self.cell
        if s1e is None:
            s1e = self.get_ovlp(cell)

        dm = xtb.GFN1XTB.get_init_guess(self, cell, key=key)

        nkpts = len(self.kpts)
        dm_kpts = np.repeat(dm[None, :, :], nkpts, axis=0)

        ne = np.einsum("kij,kji->", dm_kpts, s1e).real
        nelectron = self.tot_electrons
        # safe normalization: a fully padded (empty) cell has ne = 0
        ne_safe = np.where(ne > 1e-12, ne, 1.0)
        scale = np.where(ne > 1e-12, nelectron / ne_safe, 0.0)
        dm_kpts = dm_kpts * scale
        return dm_kpts.astype(np.complexx)

    def get_hcore(
        self,
        cell: CellPad | None = None,
        s1e: ArrayLike | None = None,
        kpts: ArrayLike | None = None,
    ) -> Array:
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts = self.kpts

        param = self.param

        mask = util.mask_valence_shell_gfn1(cell)
        hscale = np.where(
            np.outer(mask, mask),
            param.k_shlpr * param.kpair * xtb.EHT_X_GFN1(cell, param),
            param.k_shlpr,
        )

        hdiag = xtb.EHT_Hdiag_GFN1(cell, param)
        pair_mask = util.mask_atom_pairs(cell)[util.atom_to_bas_indices_2d(cell)]

        Ls = cell.Ls
        nL = len(Ls)
        h1 = np.where(
            np.repeat(pair_mask[None, :, :], nL, axis=0),
            hscale[None, :, :] * EHT_PI_GFN1(cell, param, Ls=Ls) * hdiag[None, :, :],
            np.repeat(hdiag[None, :, :], nL, axis=0),
        )

        shl_mask = self.cell.shl_mask
        shl_pair_mask = np.outer(shl_mask, shl_mask)
        h1 = np.where(shl_pair_mask[None, ...], h1, 0.0)
        h1 = np.asarray(h1, dtype=np.floatx)

        if cell is self.cell:
            s1e_lat = self.s1e_lat
        else:
            s1e_lat = self.get_ovlp_lat(cell=cell, Ls=Ls)

        expkL = np.exp(1j * np.dot(kpts, Ls.T)).astype(np.complexx)
        i, j = util.bas_to_ao_indices_2d(cell)
        hcore = np.einsum("kl,lpq->kpq", expkL, s1e_lat * h1[:, i, j])

        if cell.cuint_plan is None:
            hcore = hermi_triu(hcore)
        else:
            hcore = hcore + hcore.transpose(0, 2, 1).conj()
        return hcore
