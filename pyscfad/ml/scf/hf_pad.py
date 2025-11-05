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

from pyscfad import numpy as np
from pyscfad.scf import hf_lite as hf
from pyscfad.scipy.linalg import eigh

def get_occ(mf, mo_energy=None, mo_coeff=None):
    # NOTE assuming mo_energy is in ascending order
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    #e_sort = mo_energy
    #nmo = mo_energy.size

    nocc = mf.tot_electrons // 2

    mask_fake_ao = np.asarray(1 - mf.mol.ao_mask, dtype=bool)
    mo_coeff_fake_ao = np.where(mask_fake_ao[:,None], mo_coeff, 0)
    tmp = np.linalg.norm(mo_coeff_fake_ao, axis=0)
    # FIXME is 1e-14 okay?
    mask = np.where(np.logical_and(abs(mo_energy)<1e-14, tmp>1e-14), False, True)
    pick = (np.cumsum(mask) <= nocc) & mask
    mo_occ = np.where(pick, 2., 0.)

    #if mf.verbose >= logger.INFO and nocc < nmo:
    #    jax.lax.cond(
    #        np.greater(e_sort[nocc-1]+1e-3, e_sort[nocc]),
    #        lambda e_homo, e_lumo: logger.warn(mf, "HOMO %.15g == LUMO %.15g", e_homo, e_lumo),
    #        lambda e_homo, e_lumo: logger.info(mf, "  HOMO = %.15g  LUMO = %.15g", e_homo, e_lumo),
    #        e_sort[nocc-1], e_sort[nocc],
    #    )

    #if mf.verbose >= logger.DEBUG:
    #    numpy.set_printoptions(threshold=nmo)
    #    logger.debug(mf, "  mo_energy =\n%s", mo_energy)
    #    numpy.set_printoptions(threshold=1000)
    return mo_occ

def make_rdm1(mo_coeff, mo_occ, **kwargs):
    dm = (mo_coeff * mo_occ) @ mo_coeff.conj().T
    return dm

def get_grad(mo_coeff, mo_occ, fock_ao):
    fock_mo = mo_coeff.conj().T @ fock_ao @ mo_coeff
    mask = np.where(mo_occ > 0, True, False)
    g = 2 * fock_mo * ((1-mask[:,None]) * mask[None,:])
    return g.ravel()

class SCF(hf.SCF):
    def __init__(self, mol):
        super().__init__(mol)

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
        mask = np.asarray(1 - self.mol.ao_mask, dtype=np.int32)
        s = s + np.diag(mask)
        return eigh(h, s)

SCFPad = SCF


if __name__ == "__main__":
    import numpy
    import jax
    from pyscfad.ml.gto import basis_array, MolePad
    #from pyscfad.xtb import basis as xtb_basis
    #bfile = xtb_basis.get_basis_filename()

    import os
    from pyscf.gto import basis
    bfile = os.path.dirname(basis.__file__) + "/sto-3g.dat"
    basis = basis_array(bfile, max_number=8)

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
        dm0 = numpy.zeros((mol.nao, mol.nao))
        for i in range(5):
            dm0[i,i] = 2.

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
