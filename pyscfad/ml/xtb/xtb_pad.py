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

"""
Molecular XTB with padding
"""
from __future__ import annotations

from pyscf.gto.mole import ANG_OF

from pyscfad import numpy as np
from pyscfad.lib import logger
from pyscfad.xtb.xtb import XTB as XTBBase
from pyscfad.xtb.xtb import GFN1XTB as GFN1XTBBase
from pyscfad.xtb import util
from pyscfad.xtb.data.elements import N_VALENCE_ARRAY

from pyscfad.ml.gto import MolePad
from pyscfad.ml.scf import SCFPad
from pyscfad.ml.xtb.param import ParamArray

def tot_valence_electrons(mol, charge: int = None, nkpts: int = 1):
    if charge is None:
        charge = mol.charge

    nelecs = N_VALENCE_ARRAY[mol.numbers]
    n = np.sum(nelecs) * nkpts - charge
    return n

def dip_moment(mol, dm, unit="Debye", verbose=logger.NOTE):
    from pyscf.data import nist
    log = logger.new_logger(mol, verbose)

    ao_dip = mol.intor_symmetric("int1e_r", comp=3)
    el_dip = np.einsum("xij,ji->x", ao_dip, dm)

    charges = N_VALENCE_ARRAY[mol.numbers]
    coords  = np.asarray(mol.atom_coords())
    nucl_dip = np.einsum("i,ix->x", charges.astype(coords.dtype), coords)
    mol_dip = nucl_dip - el_dip

    if unit.upper() == "DEBYE":
        mol_dip *= nist.AU2DEBYE
        log.note("Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f", *mol_dip)
    else:
        log.note("Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f", *mol_dip)
    del log
    return mol_dip

class XTB(XTBBase, SCFPad):
    @property
    def tot_electrons(self):
        return tot_valence_electrons(self.mol)

    def dip_moment(self, mol=None, dm=None, unit="Debye", verbose=None,
                   **kwargs):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm =self.make_rdm1()
        if verbose is None:
            verbose = mol.verbose
        return dip_moment(mol, dm, unit, verbose=verbose)

    get_occ = SCFPad.get_occ

class GFN1XTB(GFN1XTBBase, XTB):
    def _get_gamma(self):
        gamma = GFN1XTBBase._get_gamma(self)

        shl_mask = self.mol.shl_mask
        mask = np.outer(shl_mask, shl_mask)
        gamma = np.where(mask, gamma, 0.)
        return gamma

    get_occ = XTB.get_occ
    dip_moment = XTB.dip_moment

if __name__ == "__main__":
    import jax
    from pyscfad.xtb import basis as xtb_basis
    from pyscfad.ml.gto import make_basis_array
    from pyscfad.ml.xtb.param import make_param_array

    bfile = xtb_basis.get_basis_filename()
    basis = make_basis_array(bfile, max_number=8)
    param = make_param_array(basis, max_number=8)

    numbers = np.array(
        [
            [8, 1, 1, 0, 0],
            [7, 1, 1, 1, 0],
        ],
        dtype=np.int32
    )
    coords = np.array(
        [
            np.array([
                [0.00000,  0.00000,  0.00000],
                [1.43355,  0.00000, -0.95296],
                [1.43355,  0.00000,  0.95296],
                [0.00000,  0.00000,  0.00000],
                [1.00000,  0.00000,  0.00000],
            ]),
            np.array([
                [-0.80650, -1.00659,  0.02850],
                [-0.50540, -0.31299,  0.68220],
                [ 0.00620, -1.41579, -0.38500],
                [-1.32340, -0.54779, -0.69350],
                [ 0.00000,  0.00000,  0.00000],
            ]) / 0.52917721067121,
        ]
    )

    def energy(numbers, coords):
        mol = MolePad(numbers, coords, basis=basis, verbose=4, trace_coords=True)
        mf = GFN1XTB(mol, param)
        mf.diis = "anderson"
        mf.conv_tol = 1e-6
        mf.diis_damp = 0.5
        mf.diis_space = 6
        #mf.sigma = 0.001
        e = mf.kernel()
        mu = mf.dip_moment()
        r2 = mol.intor("int1e_r2", hermi=1)
        e_r2 = np.einsum("ij,ij->", mf.make_rdm1(), r2)
        e_homo, e_lumo = mf.get_homo_lumo_energy()
        return e, {"dip": mu, "r2": e_r2, "e_homo": e_homo, "e_lumo": e_lumo}

    gfn = jax.value_and_grad(energy, 1, has_aux=True)
    (e, aux_res), g = jax.jit(jax.vmap(gfn))(numbers, coords)
    print(e)
    print(g)
    print(aux_res)
