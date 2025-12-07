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

"""
XTB
"""
import numpy
import jax

from pyscf.gto.mole import ANG_OF

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.lib import logger
from pyscfad.gto.mole import inter_distance
from pyscfad.scf import hf_lite as hf
from pyscfad.dft.rks import VXC

from pyscfad.xtb import util
from pyscfad.xtb.data.radii import ATOMIC as ATOMIC_RADII
from pyscfad.xtb.data.elements import N_VALENCE

def tot_valence_electrons(mol, charge=None, nkpts=1):
    if charge is None:
        charge = mol.charge

    nelecs = [N_VALENCE.get(elem) for elem in mol.elements]
    n = numpy.sum(nelecs) * nkpts - charge
    n = int(n + 0.5)
    return n

def get_occ(mf, mo_energy=None, mo_coeff=None):
    # NOTE assuming mo_energy is in ascending order
    # so that mo_occ can be made static
    if mo_energy is None:
        mo_energy = mf.mo_energy
    #e_idx = np.argsort(mo_energy)
    e_sort = mo_energy#[e_idx]
    nmo = mo_energy.size
    mo_occ = numpy.zeros(nmo)
    nocc = mf.tot_electrons // 2
    #mo_occ = ops.index_update(mo_occ, ops.index[e_idx[:nocc]], 2)
    mo_occ[:nocc] = 2
    if mf.verbose >= logger.INFO and nocc < nmo:
        jax.lax.cond(
            np.greater(e_sort[nocc-1]+1e-3, e_sort[nocc]),
            lambda e_homo, e_lumo: logger.warn(mf, "HOMO %.15g == LUMO %.15g", e_homo, e_lumo),
            lambda e_homo, e_lumo: logger.info(mf, "  HOMO = %.15g  LUMO = %.15g", e_homo, e_lumo),
            e_sort[nocc-1], e_sort[nocc],
        )

    if mf.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        logger.debug(mf, "  mo_energy =\n%s", mo_energy)
        numpy.set_printoptions(threshold=1000)
    return mo_occ

class XTB(hf.SCF):
    """Base class for XTB
    """
    init_guess = "refocc"
    #_dynamic_attr = ["_enuc", "_gamma", "param"]

    def __init__(self, mol, param=None):
        super().__init__(mol)
        if param is None:
            self.param = None
        else:
            self.param = param.to_mol_param(mol)
        self._enuc = None
        self._gamma = None

    def build(self, mol=None):
        _ = self.energy_nuc()
        _ = self.gamma

    def get_hcore(self, mol=None, **kwargs):
        raise NotImplementedError

    def get_veff(self, mol=None, dm=None, s1e=None):
        raise NotImplementedError

    def get_init_guess(self, mol=None, key="refocc", **kwargs):
        raise NotImplementedError

    def _energy_nuc(self, mol=None, **kwargs):
        raise NotImplementedError

    def energy_nuc(self):
        if self._enuc is None:
            self._enuc = self._energy_nuc()
        return self._enuc

    def _get_EHT_factor(self, mol=None, s1e=None):
        raise NotImplementedError

    def _get_gamma(self):
        raise NotImplementedError

    @property
    def gamma(self):
        if self._gamma is None:
            self._gamma = self._get_gamma()
        return self._gamma

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        if h1e is None:
            h1e = self.get_hcore(self.mol)
        if vhf is None or getattr(vhf, "ecoul", None) is None:
            vhf = self.get_veff(self.mol, dm)

        e1 = np.einsum("ij,ji", h1e, dm)
        ecoul = vhf.ecoul
        tot_e = e1 + ecoul
        return tot_e.real, ecoul

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        nuc = self.energy_nuc()
        e_tot = self.energy_elec(dm, h1e, vhf)[0] + nuc
        return e_tot

    @property
    def tot_electrons(self):
        return tot_valence_electrons(self.mol)

    get_occ = get_occ


def EHT_X_GFN1(mol, param):
    kEN = param.kEN
    EN = param.EN

    EN_AB = (EN[:,None] - EN[None,:])
    X = np.where(util.mask_atom_pairs(mol),
                 1. + kEN * EN_AB * EN_AB,
                 0.)
    return X[util.atom_to_bas_indices_2d(mol)]

def EHT_Y_GFN1(*args, **kwargs):
    return 1

def EHT_Hdiag_GFN1(mol, param):
    selfenergy = param.selfenergy
    kcn = param.kcn
    CN = param.CN[util.atom_to_bas_indices(mol)]

    Hdiag = selfenergy - kcn * CN
    return .5 * (Hdiag[:,None] + Hdiag[None,:])

def EHT_PI_GFN1(mol, param, atomic_radii=ATOMIC_RADII):
    shpoly = param.shpoly

    rr = inter_distance(mol)

    z = mol.atom_charges()
    cov_radii = atomic_radii[z]
    RAB = cov_radii[:,None] + cov_radii[None,:]

    if hasattr(mol, "atom_mask"):
        mask = np.outer(mol.atom_mask, mol.atom_mask)
        rr = np.where(mask, rr, 0.)
        RAB = np.where(mask, RAB, np.inf)

    rr = np.sqrt(rr / RAB)

    RR = rr[util.atom_to_bas_indices_2d(mol)]
    PI = (1 + shpoly[:,None] * RR) * (1 + shpoly[None,:] * RR)
    return PI

def mulliken_charge(mol, param, s1e, dm):
    SP = np.einsum("pq,pq->p", s1e, dm)
    occs = np.zeros(mol.nbas)
    occs = ops.index_add(occs, ops.index[util.bas_to_ao_indices(mol)], SP)
    return param.refocc - occs

def sum_shell_charges(mol, partial_charges):
    atm_charges = np.zeros(mol.natm)
    atm_charges = ops.index_add(
                        atm_charges,
                        ops.index[util.atom_to_bas_indices(mol)],
                        partial_charges)
    return atm_charges

def gamma_GFN1(mol, param):
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
    def _get_gamma(self):
        return gamma_GFN1(self.mol, self.param)

    def get_hcore(self, mol=None, s1e=None):
        if mol is None:
            mol = self.mol
        if s1e is None:
            s1e = self.get_ovlp()
        return s1e * self._get_EHT_factor(mol, s1e)

    def _get_EHT_factor(self, mol=None, s1e=None):
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

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1, s1e=None, **kwargs):
        del dm_last, vhf_last
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if s1e is None:
            s1e = self.get_ovlp()

        param = self.param
        partial_charge = mulliken_charge(mol, param, s1e, dm)

        phi = np.dot(self.gamma, partial_charge)
        ecoul = .5 * np.dot(partial_charge, phi)

        # Third-order term
        atm_charge = sum_shell_charges(mol, partial_charge)
        phi3 = atm_charge**2 * param.gam3
        ecoul += np.sum(atm_charge**3 * param.gam3) / 3.

        atm_to_bas_id = util.atom_to_bas_indices(mol)
        phi += phi3[atm_to_bas_id]
        phi = phi[util.bas_to_ao_indices(mol)]
        phi = phi[:,None] + phi[None,:]

        vj = -.5 * s1e * phi
        vxc = VXC(vxc=vj, ecoul=ecoul)
        return vxc

    def _energy_nuc(self, mol=None, **kwargs):
        if mol is None:
            mol = self.mol

        param = self.param
        kf = param.kf
        zeff = param.zeff
        arep = param.arep

        r = inter_distance(mol)
        r_inv = 1. / np.where(r>1e-6, r, np.inf)

        z_ab = zeff[:,None] * zeff[None,:]
        arep_ab = np.sqrt(arep[:,None] * arep[None,:])

        damp = np.where(r>1e-6, np.exp(-arep_ab * r**kf), 0)
        enuc = .5 * np.sum(z_ab * damp * r_inv)
        return enuc

    def get_init_guess(self, mol=None, key="refocc", **kwargs):
        if key != "refocc":
            raise NotImplementedError(f"Unsupported initial guess type {key}")

        if mol is None:
            mol = self.mol

        refocc = self.param.refocc / (2 * mol._bas[:,ANG_OF] + 1)
        refocc = refocc[util.bas_to_ao_indices(mol)]
        dm = np.diag(refocc)
        return dm
