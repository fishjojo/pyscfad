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

"""Tests for pyscfad.scf
"""
import numpy
import jax
from pyscfad import config_update
from pyscfad import scf
from .util import (
    hf_energy,
    df_hf_energy,
    hf_nuc_grad,
    hf_nuc_hess,
    hf_nuc_deriv3,
)

def test_rhf_nuc_grad(mol_H2O, mol_N2):
    mol = mol_H2O()
    g1 = jax.grad(hf_energy)(mol, scf.RHF).coords
    g0 = hf_nuc_grad(mol, scf.RHF)
    assert abs(g1 - g0).max() < 1e-6

    mol = mol_N2()
    g1 = jax.grad(hf_energy)(mol, scf.RHF).coords
    g0 = hf_nuc_grad(mol, scf.RHF)
    assert abs(g1 - g0).max() < 1e-6

def test_uhf_nuc_grad(mol_H2O):
    mol = mol_H2O(charge=1, spin=1)
    g1 = jax.grad(hf_energy)(mol, scf.UHF).coords
    g0 = hf_nuc_grad(mol, scf.UHF)
    assert abs(g1 - g0).max() < 1e-6

def test_rohf_nuc_grad(mol_H2O):
    mol = mol_H2O(charge=1, spin=1)
    g1 = jax.grad(hf_energy)(mol, scf.ROHF).coords
    g0 = hf_nuc_grad(mol, scf.ROHF)
    assert abs(g1 - g0).max() < 1e-6

def test_rhf_nuc_hess(mol_N2):
    mol = mol_N2()
    hess1 = jax.jacrev(jax.grad(hf_energy))(mol, scf.RHF).coords.coords
    hess0 = hf_nuc_hess(mol, scf.RHF)
    assert abs(hess1 - hess0).max() < 1e-6

def test_rhf_nuc_deriv3(mol_H2):
    mol = mol_H2(basis="sto3g")
    e3 = jax.jacfwd(jax.jacfwd(jax.grad(hf_energy)))(mol, scf.RHF).coords.coords.coords
    e3_fdiff = hf_nuc_deriv3(mol, scf.RHF)
    assert abs(e3 - e3_fdiff).max() < 1e-6

def test_df_rhf_nuc_grad(mol_H2):
    mol = mol_H2()
    g0 = numpy.array([[0,0,-0.007562138], [0,0,0.007562138]])

    g1 = jax.grad(df_hf_energy)(mol, scf.RHF).coords
    assert abs(g1 - g0).max() < 1e-6

    with config_update('pyscfad_scf_implicit_diff', True):
        g1 = jax.grad(df_hf_energy)(mol, scf.RHF).coords
    assert abs(g1 - g0).max() < 1e-6

def test_to_pyscf(mol_N2):
    ehf = scf.UHF(mol_N2()).density_fit().to_pyscf().kernel()
    assert abs(ehf - -108.867850114325) < 1e-8
