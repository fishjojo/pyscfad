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
Molecular XTB models.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy
from pyscf.gto.mole import ANG_OF

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.gto.mole import inter_distance
from pyscfad.scf.hf_lite import SCFLite
from pyscfad.dft.rks import VXC
from pyscfad.xtb import util
from pyscfad.xtb.data.radii import ATOMIC as ATOMIC_RADII
from pyscfad.xtb.data.elements import N_VALENCE

if TYPE_CHECKING:
    from typing import Any
    from pyscfad.typing import ArrayLike, Array
    from pyscfad.gto import MoleLite
    from pyscfad.xtb.param import GFN1MolParam

def tot_valence_electrons(mol: MoleLite, charge: int | None = None, nkpts: int = 1) -> Array:
    if charge is None:
        charge = mol.charge

    nelecs = [N_VALENCE.get(elem) for elem in mol.elements]
    n = numpy.sum(nelecs) * nkpts - charge
    return n

class XTB(ABC, SCFLite):
    """Base class for XTB methods.
    """
    init_guess = "refocc"
    veff_with_ecoul = True

    def __init__(self, mol: MoleLite, param: Any | None = None, **kwargs):
        if hasattr(param, "to_mol_param"):
            self.param = param.to_mol_param(mol)
        else:
            self.param = param
        self._enuc = None
        self._gamma = None

        super().__init__(mol, **kwargs)

    def build(self, mol: MoleLite | None = None) -> XTB:
        # cache quantities that are computed only once
        _ = self.energy_nuc(mol)
        _ = self.gamma
        return self

    @abstractmethod
    def get_hcore(
        self,
        mol: MoleLite | None = None,
        s1e: ArrayLike | None = None,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def get_veff(
        self,
        mol: MoleLite | None = None,
        dm: ArrayLike | None = None,
        dm_last: ArrayLike = np.array(0.),
        vhf_last: ArrayLike = np.array(0.),
        hermi: int = 1,
        s1e: ArrayLike | None = None,
        q: ArrayLike | None = None,
        **kwargs,
    ) -> VXC:
        raise NotImplementedError

    @abstractmethod
    def get_init_guess(
        self,
        mol: MoleLite | None = None,
        key: str = "refocc",
        **kwargs
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _energy_nuc(self, mol: MoleLite | None = None, **kwargs) -> float:
        raise NotImplementedError

    def energy_nuc(self, mol: MoleLite | None = None) -> float:
        if self._enuc is None:
            self._enuc = self._energy_nuc(mol)
        return self._enuc

    @abstractmethod
    def _get_EHT_factor(
        self,
        mol: MoleLite | None = None,
        s1e: ArrayLike | None = None,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _get_gamma(self) -> Array:
        raise NotImplementedError

    @property
    def gamma(self) -> Array:
        if self._gamma is None:
            self._gamma = self._get_gamma()
        return self._gamma

    def scf(self, dm0: ArrayLike | None = None, q0: ArrayLike | None = None, **kwargs) -> float:
        if self.diis == "qbroyden":
            from pyscfad.xtb import qbroyden

            self.dump_flags()
            self.build(self.mol)

            if self.max_cycle > 0 or self.mo_coeff is None:
                self.converged, self.e_tot, \
                        self.mo_energy, self.mo_coeff, self.mo_occ = \
                        qbroyden.scf(self,
                                     conv_tol=self.conv_tol,
                                     conv_tol_grad=self.conv_tol_grad,
                                     dm0=dm0, q0=q0)
            else:
                self.e_tot = qbroyden.scf(self,
                                          conv_tol=self.conv_tol,
                                          conv_tol_grad=self.conv_tol_grad,
                                          dm0=dm0, q0=q0)[1]
            return self.e_tot

        else:
            return super().scf(dm0=dm0, **kwargs)

    @property
    def tot_electrons(self) -> int:
        return tot_valence_electrons(self.mol)

    def dip_moment(
        self,
        mol: MoleLite | None = None,
        dm: ArrayLike | None = None,
        unit: str = "Debye",
        origin: ArrayLike | None = None,
        verbose: int | None = None,
        charges: ArrayLike | None = None,
    ) -> Array:
        if mol is None:
            mol = self.mol
        if charges is None:
            charges = np.asarray([N_VALENCE.get(elem) for elem in mol.elements])
        return super().dip_moment(mol=mol, dm=dm, unit=unit, origin=origin, verbose=verbose,
                                  charges=charges)

    def shell_charges(
        self,
        mol: MoleLite | None = None,
        dm: ArrayLike | None = None,
        s1e: ArrayLike | None = None,
        method: str = "mulliken"
    ) -> Array:
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if s1e is None:
            s1e = self.get_ovlp()

        if method.lower() == "mulliken":
            q = mulliken_charge(mol, self.param, s1e, dm)
        else:
            raise NotImplementedError
        return q
    get_q = shell_charges

def EHT_X_GFN1(mol: MoleLite, param: GFN1MolParam) -> Array:
    kEN = param.kEN
    EN = param.EN

    EN_AB = (EN[:,None] - EN[None,:])
    X = np.where(util.mask_atom_pairs(mol),
                 1. + kEN * EN_AB * EN_AB,
                 0.)
    return X[util.atom_to_bas_indices_2d(mol)]

def EHT_Y_GFN1(*args, **kwargs):
    return 1

def EHT_Hdiag_GFN1(mol: MoleLite, param: GFN1MolParam) -> Array:
    selfenergy = param.selfenergy
    kcn = param.kcn
    CN = param.CN[util.atom_to_bas_indices(mol)]

    Hdiag = selfenergy - kcn * CN
    return .5 * (Hdiag[:,None] + Hdiag[None,:])

def EHT_PI_GFN1(
    mol: MoleLite,
    param: GFN1MolParam,
    atomic_radii: ArrayLike = ATOMIC_RADII,
) -> Array:
    shpoly = param.shpoly

    rr = inter_distance(mol)

    z = mol.atom_charges()
    cov_radii = atomic_radii[z]
    RAB = cov_radii[:,None] + cov_radii[None,:]

    if hasattr(mol, "atom_mask"):
        mask = np.outer(mol.atom_mask, mol.atom_mask)
        rr = np.where(mask, rr, 0.)
        RAB = np.where(mask, RAB, np.inf)

    rr = np.safe_sqrt(rr / RAB, thresh=1e-6)

    RR = rr[util.atom_to_bas_indices_2d(mol)]
    PI = (1 + shpoly[:,None] * RR) * (1 + shpoly[None,:] * RR)
    return PI

def mulliken_charge(
    mol: MoleLite,
    param: Any,
    s1e: ArrayLike,
    dm: ArrayLike,
) -> Array:
    PS = np.einsum("pq,qp->p", dm, s1e)
    occs = np.zeros(mol.nbas)
    occs = ops.index_add(occs, ops.index[util.bas_to_ao_indices(mol)], PS)
    return param.refocc - occs

def sum_shell_charges(mol: MoleLite, partial_charges: ArrayLike) -> Array:
    atm_charges = np.zeros(mol.natm)
    atm_charges = ops.index_add(
                        atm_charges,
                        ops.index[util.atom_to_bas_indices(mol)],
                        partial_charges)
    return atm_charges

def gamma_GFN1(mol: MoleLite, param: GFN1MolParam) -> Array:
    eta = 1. / (param.lgam * param.gam)
    eta = 2. / (eta[:,None] + eta[None,:])

    r = inter_distance(mol)
    i, j = util.atom_to_bas_indices_2d(mol)
    r = r[i,j]

    gamma = np.where(r<1e-6, eta, np.sqrt(1./(r**2 + 1./eta**2)))
    return gamma


class GFN1XTB(XTB):
    """GFN1-XTB
    """
    def _get_gamma(self) -> Array:
        return gamma_GFN1(self.mol, self.param)

    def get_hcore(self, mol: MoleLite | None = None, s1e: ArrayLike | None = None) -> Array:
        if mol is None:
            mol = self.mol
        if s1e is None:
            s1e = self.get_ovlp()
        return s1e * self._get_EHT_factor(mol, s1e)

    def _get_EHT_factor(self, mol: MoleLite | None = None, s1e: ArrayLike | None = None) -> Array:
        if mol is None:
            mol = self.mol
        if s1e is None:
            s1e = self.get_ovlp()

        param = self.param

        mask = util.mask_valence_shell_gfn1(mol)
        hscale = np.where(np.outer(mask, mask),
                          param.k_shlpr * param.kpair * EHT_X_GFN1(mol, param),
                          param.k_shlpr)

        hdiag = EHT_Hdiag_GFN1(mol, param)
        mask = util.mask_atom_pairs(mol)[util.atom_to_bas_indices_2d(mol)]
        h1 = np.where(mask,
                      hscale * EHT_PI_GFN1(mol, param) * hdiag,
                      hdiag)
        return h1[util.bas_to_ao_indices_2d(mol)]

    def get_veff(
        self,
        mol: MoleLite | None = None,
        dm: ArrayLike | None = None,
        dm_last: ArrayLike = np.array(0.),
        vhf_last: ArrayLike = np.array(0.),
        hermi: int = 1,
        s1e: ArrayLike | None = None,
        q: ArrayLike | None = None,
        **kwargs,
    ) -> VXC:
        del dm_last, vhf_last
        if mol is None:
            mol = self.mol
        if s1e is None:
            s1e = self.get_ovlp()
        if q is None:
            q = self.get_q(mol=mol, dm=dm, s1e=s1e)

        param = self.param

        mono = q[:mol.nbas]
        phi = np.dot(self.gamma, mono)
        ecoul = .5 * np.dot(mono, phi)

        # Third-order term
        atm_charge = sum_shell_charges(mol, mono)
        phi3 = atm_charge**2 * param.gam3
        ecoul += np.sum(atm_charge**3 * param.gam3) / 3.

        atm_to_bas_id = util.atom_to_bas_indices(mol)
        phi += phi3[atm_to_bas_id]
        phi = phi[util.bas_to_ao_indices(mol)]
        phi = phi[:,None] + phi[None,:]

        vj = -.5 * s1e * phi
        return VXC(vxc=vj, ecoul=ecoul)

    def _energy_nuc(self, mol: MoleLite | None = None, **kwargs) -> float:
        if mol is None:
            mol = self.mol

        param = self.param
        kf = param.kf
        zeff = param.zeff
        arep = param.arep

        r, r_inv = util.r_and_inv_r(mol)
        r_safe = np.where(r>1e-6, r, 1.0)

        z_ab = zeff[:,None] * zeff[None,:]
        arep_ab = np.safe_sqrt(arep[:,None] * arep[None,:], thresh=1e-14)

        damp = np.where(r>1e-6, np.exp(-arep_ab * r_safe**kf), 0.)
        enuc = .5 * np.sum(z_ab * damp * r_inv)
        return enuc

    def get_init_guess(self, mol: MoleLite | None = None, key: str = "refocc", **kwargs) -> Array:
        if key != "refocc":
            raise NotImplementedError(f"Unsupported initial guess type {key}")

        if mol is None:
            mol = self.mol

        refocc = self.param.refocc / (2 * mol._bas[:,ANG_OF] + 1)
        refocc = refocc[util.bas_to_ao_indices(mol)]
        dm = np.diag(refocc)
        return dm
