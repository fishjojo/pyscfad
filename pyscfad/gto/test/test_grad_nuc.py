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

import pytest
import jax
import pyscf
from pyscf.grad.rhf import grad_nuc
from pyscf.hessian.rhf import hess_nuc
from pyscfad import gto

@pytest.fixture
def get_mol0():
    mol = pyscf.M(
        atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587',
        verbose=0,
    )
    return mol

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.verbose=0
    mol.build(trace_coords=True)
    return mol

# pylint: disable=redefined-outer-name
def test_grad_nuc(get_mol0, get_mol):
    mol0 = get_mol0
    g0 = grad_nuc(mol0)
    mol = get_mol
    jac = jax.grad(mol.__class__.energy_nuc)(mol)
    g = jac.coords
    assert abs(g-g0).max() < 1e-10

def test_hess_nuc(get_mol0, get_mol):
    mol0 = get_mol0
    h0 = hess_nuc(mol0).transpose(0,2,1,3)

    mol = get_mol
    hess = jax.jacfwd(jax.grad(mol.__class__.energy_nuc))(mol)
    h = hess.coords.coords
    assert abs(h-h0).max() < 1e-10
