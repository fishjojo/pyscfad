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

import numpy as np
from jax import jacfwd
from pyscf.lib.parameters import BOHR
from pyscf import scf as pyscf_scf
from pyscfad import gto
from pyscfad import scf

def build_mol(atom, unit="A"):
    mol = gto.Mole()
    mol.atom = atom
    mol.unit = unit
    mol.basis = 'sto3g'
    mol.build(trace_coords=True, trace_exp=False, trace_ctr_coeff=False)
    return mol

def hess_anl(mol):
    mf = pyscf_scf.RHF(mol)
    mf.kernel()
    hess = mf.Hessian().kernel()
    return hess.transpose(0,2,1,3)

def get_fdiff_mol(mol, d, unit="A"):
    res = []
    coords = np.asarray(mol.atom_coords(unit=unit))
    for ia in range(mol.natm):
        for x in range(3):
            for sign in (1, -1):
                new_coords = coords.copy()
                new_coords[ia, x] += sign * d
                new_atoms = list(zip([atom[0] for atom in mol._atom], new_coords.tolist()))
                res.append(build_mol(new_atoms, unit=unit))
    return res

def derivative(plus, minus, d, unit="A"):
    if unit == "A":
        d /= BOHR
    return (plus - minus) / (2 * d)

def get_finite_derivatives(finite_diff, d, unit="A"):
    deriv_list = []
    for i in range(0, len(finite_diff), 2):
        deriv = derivative(finite_diff[i], finite_diff[i+1], d, unit=unit)
        deriv_list.append(deriv)
    return np.array(deriv_list)

def test_deriv3():
    def ehf(mol):
        mf = scf.RHF(mol)
        e = mf.kernel()
        return e

    atom = "H 0 0 -0.5; H 0 0 0.5"
    mol = build_mol(atom)
    e3 = jacfwd(jacfwd(jacfwd(ehf)))(mol).coords.coords.coords

    d = 1e-5
    fdiff_mols = get_fdiff_mol(mol, d)
    e3_fdiff_list = []
    for _mol in fdiff_mols:
        e3_fdiff_list.append(hess_anl(_mol))
    e3_fdiff = get_finite_derivatives(e3_fdiff_list, d).reshape(2,3,2,3,2,3).transpose(4,5,2,3,0,1)

    assert abs(e3 - e3_fdiff).max() < 1e-6
