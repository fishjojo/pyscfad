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

"""Utility functions
"""
import numpy
import jax
from pyscf.data.nist import BOHR
from pyscf.hessian.rhf import Hessian
from pyscfad import gto

def make_mol(
    atom,
    basis = "631G",
    charge = 0,
    spin = 0,
    unit = "A",
    verbose = 0,
    max_memory = 7000,
    incore_anyway = True,
    trace_coords = True,
    trace_exp = False,
    trace_ctr_coeff = False,
):
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.unit = unit
    mol.max_memory = max_memory
    mol.verbose = verbose
    mol.incore_anyway = incore_anyway
    mol.build(
        trace_coords=trace_coords,
        trace_exp=trace_exp,
        trace_ctr_coeff=trace_ctr_coeff,
    )
    return mol

hf_energy = lambda mol, method: method(mol).kernel()
df_hf_energy = lambda mol, method: method(mol).density_fit().kernel()

def hf_nuc_grad(mol, method):
    mf = method(mol).to_pyscf()
    mf.kernel()
    grad = mf.nuc_grad_method().kernel()
    return grad

def hf_nuc_hess(mol, method):
    mf = method(mol).to_pyscf()
    mf.kernel()
    hess = Hessian(mf).kernel().transpose(0,2,1,3)
    return hess

def _get_fdiff_mols(mol, disp, unit="A"):
    mols = []
    coords = numpy.asarray(mol.atom_coords(unit=unit))
    for ia in range(mol.natm):
        for x in range(3):
            for sign in (1, -1):
                new_coords = coords.copy()
                new_coords[ia, x] += sign * disp
                new_atoms = list(zip([atom[0] for atom in mol._atom], new_coords.tolist()))
                mols.append(mol.set_geom_(new_atoms, unit=unit, inplace=False))
    return mols

def _get_finite_derivatives(finite_diff, disp, unit="A"):
    def derivative(plus, minus, d, unit="A"):
        if unit == "A":
            d /= BOHR
        return (plus - minus) / (2 * d)

    deriv_list = []
    for i in range(0, len(finite_diff), 2):
        deriv = derivative(finite_diff[i], finite_diff[i+1], disp, unit=unit)
        deriv_list.append(deriv)
    return numpy.asarray(deriv_list)

def hf_nuc_deriv3(mol, method, disp=1e-4):
    fdiff_mols = _get_fdiff_mols(mol, disp)
    e3_fdiff_list = []
    for _mol in fdiff_mols:
        e3_fdiff_list.append(hf_nuc_hess(_mol, method))
    e3_fdiff = _get_finite_derivatives(e3_fdiff_list, disp).reshape(2,3,2,3,2,3).transpose(4,5,2,3,0,1)
    return e3_fdiff

