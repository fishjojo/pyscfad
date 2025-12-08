# Copyright 2021-2025 The PySCFAD Authors
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

import numpy
from pyscfad import numpy as np
from pyscfad.lib import logger
from pyscfad.scf import hf_lite as hf
from pyscfad.scipy.linalg import eigh
from pyscfad.ops import stop_grad

def _fermi_smearing_occ(mu, mo_energy, sigma, mo_mask):
    de = (mo_energy - mu) / sigma
    de = np.where(np.less(de, 40.), de, np.inf)
    occ = np.where(np.less(de, 40.), 1. / (np.exp(de) + 1.), 0.)
    occ = np.where(mo_mask, occ, 0)
    return occ

def _smearing_optimize(mo_es, nocc, sigma, mo_mask):
    from jax.scipy import optimize
    def nelec_cost_fn(mu):
        mo_occ = _fermi_smearing_occ(mu, mo_es, sigma, mo_mask)
        return (np.sum(mo_occ) - nocc)**2

    mu0 = np.array([mo_es[nocc-1],])
    res = optimize.minimize(
        nelec_cost_fn, mu0, method="BFGS", tol=1e-5,
    )
    mu = res.x
    mo_occs = _fermi_smearing_occ(mu, mo_es, sigma, mo_mask)
    return mu, mo_occs

def make_mo_mask(mo_energy, mo_coeff, ao_mask):
    mask_fake_ao = np.asarray(1 - ao_mask, dtype=bool)
    mo_coeff_fake_ao = np.where(mask_fake_ao[:,None], mo_coeff, 0)
    tmp = np.linalg.norm(mo_coeff_fake_ao, axis=0)
    mask = np.where(np.logical_and(mo_energy>1e8, tmp>1e-12), False, True)
    return mask

def get_occ(mf, mo_energy=None, mo_coeff=None):
    # NOTE assuming mo_energy is in ascending order
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    nmo = mo_energy.size
    #e_sort = mo_energy

    mask = make_mo_mask(mo_energy, mo_coeff, mf.mol.ao_mask)
    nocc = mf.tot_electrons // 2

    if mf.sigma is not None and mf.sigma > 0:
        mu, mo_occ = _smearing_optimize(stop_grad(mo_energy), nocc, mf.sigma, mask)
        mo_occ *= 2
    else:
        pick = (np.cumsum(mask) <= nocc) & mask
        mo_occ = np.where(pick, 2., 0.)

        e_homo = np.max(np.where(pick, mo_energy, -np.inf))
        e_lumo = np.min(np.where(mask & ~pick, mo_energy, np.inf))
        if mf.verbose >= logger.DEBUG:
            logger.debug(mf, "  HOMO = %.15g  LUMO = %.15g", e_homo, e_lumo)

    if mf.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        logger.debug(mf, "  mo_energy =\n%s", mo_energy)
        numpy.set_printoptions(threshold=1000)
    return mo_occ

def get_homo_lumo_energy(mf, mo_energy=None, mo_coeff=None):
    if mf.sigma is not None and mf.sigma > 0:
        raise NotImplementedError

    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    mask = make_mo_mask(mo_energy, mo_coeff, mf.mol.ao_mask)
    nocc = mf.tot_electrons // 2
    pick = (np.cumsum(mask) <= nocc) & mask

    e_homo = np.max(np.where(pick, mo_energy, -np.inf))
    e_lumo = np.min(np.where(mask & ~pick, mo_energy, np.inf))
    if mf.verbose >= logger.DEBUG:
        logger.debug(mf, "  HOMO = %.15g  LUMO = %.15g", e_homo, e_lumo)
    return e_homo, e_lumo

def make_rdm1(mo_coeff, mo_occ, **kwargs):
    dm = (mo_coeff * mo_occ) @ mo_coeff.conj().T
    return dm

def get_grad(mo_coeff, mo_occ, fock_ao):
    fock_mo = mo_coeff.conj().T @ fock_ao @ mo_coeff
    mask = np.where(mo_occ > 0, True, False)
    g = 2 * fock_mo * ((1-mask[:,None]) * mask[None,:])
    return g.ravel()

class SCF(hf.SCF):
    def __init__(self, mol, **kwargs):
        super().__init__(mol, **kwargs)
        self.sigma = None

    def get_occ(self, mo_energy=None, mo_coeff=None):
        return get_occ(self, mo_energy=mo_energy, mo_coeff=mo_coeff)

    @property
    def tot_electrons(self):
        return self.mol.tot_electrons()

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        return make_rdm1(mo_coeff, mo_occ, **kwargs)

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)

    def _eigh(self, h, s):
        ao_mask = self.mol.ao_mask
        mask = np.asarray(1 - ao_mask, dtype=np.int32)
        s = s + np.diag(mask)
        h = np.where(np.outer(ao_mask, ao_mask), h, 0.)
        h1e_diag = np.diag(h)
        h1e_diag = np.where(ao_mask, h1e_diag, 1e10)
        h = np.fill_diagonal(h, h1e_diag, inplace=False)
        return eigh(h, s)

    get_homo_lumo_energy = get_homo_lumo_energy

SCFPad = SCF


if __name__ == "__main__":
    import numpy
    import jax
    from pyscfad.ml.gto import make_basis_array, MolePad
    #from pyscfad.xtb import basis as xtb_basis
    #bfile = xtb_basis.get_basis_filename()

    import os
    from pyscf.gto import basis
    bfile = os.path.dirname(basis.__file__) + "/sto-3g.dat"
    basis = make_basis_array(bfile, max_number=8)

    numbers = np.array([8, 1, 1, 0], dtype=np.int32)
    coords = np.array(
        [
            [0.00000,  0.00000,  0.00000],
            [1.43355,  0.00000, -0.95296],
            [1.43355,  0.00000,  0.95296],
            [0.00000,  0.00000,  0.00000],
        ]
    )

    @jax.jit
    def energy(numbers, coords):
        mol = MolePad(numbers, coords, basis=basis, verbose=4)
        dm0 = np.zeros((mol.nao, mol.nao))
        for i in range(5):
            dm0 = dm0.at[i,i].set(2.)
        mf = SCFPad(mol)
        ehf = mf.kernel(dm0=dm0)
        return ehf

    ehf = energy(numbers, coords)
    print(ehf)
    print(energy._cache_size())

    numbers = np.array([7, 1, 1, 1], dtype=np.int32)
    coords = np.array(
        [
            [-0.80650,       -1.00659,        0.02850],
            [-0.50540,       -0.31299,        0.68220],
            [ 0.00620,       -1.41579,       -0.38500],
            [-1.32340,       -0.54779,       -0.69350],
        ]
    ) / 0.52917721067121

    ehf = energy(numbers, coords)
    print(ehf)
    print(energy._cache_size())
