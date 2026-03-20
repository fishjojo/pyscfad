# Copyright 2025-2026 The PySCFAD Authors
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
XTB with k-point sampling
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from jax.scipy.special import erfc

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.lib import hermi_triu
from pyscfad.gto.mole import inter_distance
from pyscfad.pbc.scf.khf_lite import KSCFLite
from pyscfad.dft.rks import VXC
from pyscfad.pbc.tools import nimgs_to_lattice_Ls

from pyscfad.xtb import xtb
from pyscfad.xtb import util
from pyscfad.xtb.data.radii import ATOMIC as ATOMIC_RADII

if TYPE_CHECKING:
    from typing import Any
    from pyscfad.typing import ArrayLike, Array
    from pyscfad.pbc.gto import CellLite as Cell
    from pyscfad.xtb.param import GFN1MolParam

def EHT_PI_GFN1(
    cell: Cell,
    param: Any,
    atomic_radii: ArrayLike = ATOMIC_RADII,
    Ls: ArrayLike = np.zeros((1,3)),
):
    shpoly = param.shpoly

    rr = inter_distance(cell, Ls=Ls)
    #rr = np.where(rr>1e-6, rr, 0)

    z = cell.atom_charges()
    cov_radii = atomic_radii[z]
    RAB = cov_radii[:,None] + cov_radii[None,:]
    #rr = np.where(rr>1e-6, np.sqrt(rr / RAB), 0)
    rr = np.sqrt(rr / RAB[None,:,:])

    i, j = util.atom_to_bas_indices_2d(cell)
    RR = rr[:,i,j]
    PI = (1 + shpoly[None,:,None] * RR) * (1 + shpoly[None,None,:] * RR)
    return PI

def mulliken_charge(
    cell: Cell,
    param: Any,
    s1e: ArrayLike,
    dm: ArrayLike,
) -> Array:
    assert s1e.ndim == 3
    assert dm.ndim == 3

    nkpts = s1e.shape[0]
    weights = 1. / nkpts

    PS = np.einsum("kpq,kqp->p", dm, s1e).real
    occs = np.zeros(cell.nbas)
    occs = ops.index_add(occs, ops.index[util.bas_to_ao_indices(cell)], PS)
    occs *= weights
    return param.refocc - occs

def gamma_GFN1(
    cell: Cell,
    param: GFN1MolParam,
    ewald_alpha: float,
    ewald_mesh: ArrayLike | None = None,
) -> Array:
    gamma_sr = _gamma_sr_GFN1(cell, param)
    gamma_ewald = _gamma_ewald(cell, ewald_alpha, ewald_mesh)
    gamma = gamma_sr + gamma_ewald
    return gamma

def _gamma_sr_GFN1(cell: Cell, param: GFN1MolParam, rsmooth: float = 1.0) -> Array:
    eta = 1. / (param.lgam * param.gam)
    eta = 2. / (eta[:,None] + eta[None,:])

    Ls = nimgs_to_lattice_Ls(cell)
    r, r_inv = util.r_and_inv_r(cell, Ls=Ls)

    i, j = util.atom_to_bas_indices_2d(cell)
    r = r[:,i,j]
    r_inv = r_inv[:,i,j]

    rcut = cell.rcut
    _val = np.sqrt(1./(r**2 + 1./eta**2)) - r_inv
    r1 = r - (rcut - rsmooth)
    x = r1 / rsmooth
    fcut = -6. * x**5 + 15. * x**4 - 10. * x**3 + 1.

    gamma_latt = np.where(
        r < rcut,
        np.where(r<rcut-rsmooth, _val, fcut*_val),
        0,
    )
    gamma = np.sum(gamma_latt, axis=0)
    return gamma

def _gamma_sr_erfc(cell: Cell, param: GFN1MolParam) -> Array:
    eta = 1. / (param.lgam * param.gam)
    eta = 2. / (eta[:,None] + eta[None,:])
    alpha = .5 * np.sqrt(np.pi) * eta

    Ls = nimgs_to_lattice_Ls(cell)
    r, r_inv = util.r_and_inv_r(cell, Ls=Ls)
    rcut = util.rcut_erfc_over_r(alpha, cell.precision)

    i, j = util.atom_to_bas_indices_2d(cell)
    r = r[:,i,j]
    r_inv = r_inv[:,i,j]

    gamma_latt = np.where(
        r < rcut,
        -erfc(alpha[None,:,:] * r) * r_inv,
        0,
    )
    gamma = np.sum(gamma_latt, axis=0)
    gamma += np.eye(cell.natm)[i,j] * eta
    return gamma

def _gamma_ewald(
    cell: Cell,
    ewald_alpha: float,
    ewald_mesh: ArrayLike | None = None
) -> Array:
    # 1/r
    Ls = nimgs_to_lattice_Ls(cell)
    r, r_inv = util.r_and_inv_r(cell, Ls=Ls)

    ew_eta = ewald_alpha
    ew_cut = util.rcut_erfc_over_r(ew_eta, cell.precision)

    gamma_latt = np.where(
        r < ew_cut,
        erfc(ew_eta * r) * r_inv,
        0,
    )
    gamma = np.sum(gamma_latt, axis=0)

    # self energy
    gamma += np.eye(cell.natm) * (-2 * ew_eta / np.sqrt(np.pi))

    if ewald_mesh is None:
        ke_cutoff = util.ke_cutoff_ewald(
            ew_eta,
            cell.precision * cell.vol,
        )
        mesh = cell.cutoff_to_mesh(ke_cutoff)
    else:
        mesh = ewald_mesh
    Gv, _, weights = cell.get_Gv_weights(mesh)
    absG2 = np.einsum("gi,gi->g", Gv, Gv)
    absG2 = np.where(absG2==0., np.inf, absG2)

    coulG = 4 * np.pi / absG2
    coulG *= weights

    SI = np.exp(-1j * (cell.atom_coords() @ Gv.T))
    expG2 = SI * np.exp(-absG2 / (4 * ew_eta**2))
    gamma += np.einsum("ag,cg,g->ac", SI.conj(), expG2, coulG).real

    i, j = util.atom_to_bas_indices_2d(cell)
    gamma = gamma[i,j]
    return gamma


class KXTB(xtb.XTB, KSCFLite):
    """Base class for XTB methods with k-point sampling.
    """
    def __init__(
        self,
        cell: Cell,
        param: Any | None = None,
        kpts: ArrayLike | None = None,
        **kwargs,
    ):
        super().__init__(cell, param=param, kpts=kpts)

    @property
    def tot_electrons(self) -> Array:
        return xtb.tot_valence_electrons(self.cell, nkpts=len(self.kpts))

    def shell_charges(
        self,
        cell: Cell | None = None,
        dm: ArrayLike | None = None,
        s1e: ArrayLike | None = None,
        method: str = "mulliken"
    ) -> Array:
        if cell is None:
            cell = self.cell
        if dm is None:
            dm = self.make_rdm1()
        if s1e is None:
            s1e = self.get_ovlp(cell)

        if method.lower() == "mulliken":
            q = mulliken_charge(cell, self.param, s1e, dm)
        else:
            raise NotImplementedError
        return q
    get_q = shell_charges

def energy_nuc_GFN1(
    cell: Cell,
    param: GFN1MolParam,
) -> float:
    kf   = param.kf
    zeff = param.zeff
    arep = param.arep

    Ls = nimgs_to_lattice_Ls(cell)
    r, r_inv = util.r_and_inv_r(cell, Ls=Ls)
    rcut = util.rcut_enuc_GFN1(kf, zeff, arep, cell.precision)

    z_ab = zeff[:,None] * zeff[None,:]
    arep_ab = np.sqrt(arep[:,None] * arep[None,:])
    enuc_ab = np.where(
        r < rcut,
        z_ab[None,...] * np.exp(-arep_ab[None,...] * r**kf) * r_inv,
        0.,
    )
    enuc = .5 * np.sum(enuc_ab)
    return enuc

class GFN1KXTB(KXTB, xtb.GFN1XTB):
    """GFN1-XTB with k-point sampling
    """
    ewald_alpha: float = 0.4 # rcut ~ 5 Angstrom
    ewald_mesh: ArrayLike | None = None

    def _energy_nuc(self, cell: Cell | None = None) -> float:
        if cell is None:
            cell = self.cell
        enuc = energy_nuc_GFN1(cell, self.param)
        return enuc

    def _get_gamma(self) -> Array:
        return gamma_GFN1(self.cell, self.param, self.ewald_alpha, self.ewald_mesh)

    def get_init_guess(
        self,
        cell: Cell | None = None,
        key: str = "refocc",
        s1e: ArrayLike | None = None,
    ) -> Array:
        if cell is None:
            cell = self.cell
        if s1e is None:
            s1e = self.get_ovlp(cell)

        dm = super().get_init_guess(cell, key=key)

        nkpts = len(self.kpts)
        dm_kpts = np.repeat(dm[None,:,:], nkpts, axis=0)

        ne = np.einsum("kij,kji->", dm_kpts, s1e).real
        nelectron = self.tot_electrons
        dm_kpts *= (nelectron / ne).reshape(-1,1,1)
        return dm_kpts.astype(np.complex128)

    def get_hcore(
        self,
        cell: Cell | None = None,
        s1e: ArrayLike | None = None,
        kpts: ArrayLike | None = None,
    ) -> Array:
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts = self.kpts

        param = self.param

        mask = util.mask_valence_shell_gfn1(cell)
        hscale = np.where(np.outer(mask, mask),
                          param.k_shlpr * param.kpair * xtb.EHT_X_GFN1(cell, param),
                          param.k_shlpr)

        hdiag = xtb.EHT_Hdiag_GFN1(cell, param)
        mask = util.mask_atom_pairs(cell)[util.atom_to_bas_indices_2d(cell)]

        Ls = nimgs_to_lattice_Ls(cell)
        nL = len(Ls)
        h1 = np.where(np.repeat(mask[None,:,:], nL, axis=0),
                      hscale[None,:,:] * EHT_PI_GFN1(cell, param, Ls=Ls) * hdiag[None,:,:],
                      np.repeat(hdiag[None,:,:], nL, axis=0))

        s1e = cell.lattice_intor("int1e_ovlp", hermi=1, Ls=Ls)

        expkL = np.exp(1j*np.dot(kpts, Ls.T))
        i, j = util.bas_to_ao_indices_2d(cell)
        hcore = np.einsum("kl,lpq->kpq", expkL, s1e * h1[:,i,j])
        hcore = hermi_triu(hcore)
        return hcore

    def get_veff(
        self,
        cell: Cell | None = None,
        dm: ArrayLike | None = None,
        dm_last: ArrayLike = np.array(0.),
        vhf_last: ArrayLike = np.array(0.),
        hermi: int = 1,
        s1e: ArrayLike | None = None,
        kpts: ArrayLike | None = None,
        kpts_band: ArrayLike | None = None,
        q: ArrayLike | None = None,
        **kwargs,
    ) -> VXC:
        del dm_last, vhf_last
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts = self.kpts
        if s1e is None:
            s1e = self.get_ovlp(cell, kpts=kpts) #pylint: disable=E1123
        if q is None:
            q = self.get_q(cell=cell, dm=dm, s1e=s1e)

        param = self.param

        mono = q[:cell.nbas]
        phi = np.dot(self.gamma, mono)
        ecoul = .5 * np.dot(mono, phi)

        if cell.charge != 0:
            Q = cell.charge / len(kpts)
            #Q = np.sum(mono)
            ecoul += -.5 * np.pi/(self.ewald_alpha**2 * cell.vol) * Q**2
            #phi += -np.pi/(self.ewald_alpha**2 * cell.vol) * Q

        # Third-order term
        atm_charge = xtb.sum_shell_charges(cell, mono)
        phi3 = atm_charge**2 * param.gam3
        ecoul += np.sum(atm_charge**3 * param.gam3) / 3.

        atm_to_bas_id = util.atom_to_bas_indices(cell)
        phi += phi3[atm_to_bas_id]
        phi = phi[util.bas_to_ao_indices(cell)]
        phi = phi[:,None] + phi[None,:]

        vj = -.5 * s1e * phi[None,...]
        vxc = VXC(vxc=vj, ecoul=ecoul)
        return vxc

