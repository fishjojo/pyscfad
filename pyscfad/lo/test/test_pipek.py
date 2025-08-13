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

import pytest
import jax
from pyscf.data.nist import BOHR
from pyscfad import numpy as np
from pyscfad import gto, scf
from pyscfad.lo import pipek
from pyscfad import config_update

def cost_function(mol, pop_method, exponent):
    mf = scf.RHF(mol)
    mf.kernel()
    orbocc = mf.mo_coeff[:,mf.mo_occ>0]
    mo_coeff = pipek.pm(mol, orbocc, conv_tol=1e-10, init_guess='atomic')

    pop = pipek.atomic_pops(mol, mo_coeff, pop_method)
    return -(np.einsum('xii->xi', pop)**exponent).sum()

@pytest.fixture
def get_mol():
    with config_update('pyscfad_scf_implicit_diff', True):
        mol = gto.Mole()
        mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
        mol.basis = '631G'
        mol.verbose = 0
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        yield mol

def _cost_nuc_grad(mol, pop_method, exponent):
    g0 = jax.grad(cost_function)(mol, pop_method, exponent).coords

    mol.set_geom_('O 0. 0.  0.0001; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587')
    f1 = cost_function(mol, pop_method, exponent)
    mol.set_geom_('O 0. 0. -0.0001; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587')
    f2 = cost_function(mol, pop_method, exponent)
    g1 = (f1 - f2) / (0.0002 / BOHR)
    assert abs(g0[0,2]-g1) < 1e-4

    mol.set_geom_('O 0. 0. 0.; H 0. , -0.756 , 0.587; H 0. , 0.757 , 0.587')
    f1 = cost_function(mol, pop_method, exponent)
    mol.set_geom_('O 0. 0. 0.; H 0. , -0.758 , 0.587; H 0. , 0.757 , 0.587')
    f2 = cost_function(mol, pop_method, exponent)
    g1 = (f1 - f2) / (0.002 / BOHR)
    assert abs(g0[1,1]-g1) < 1e-4

    mol.set_geom_('O 0. 0. 0.; H 0. , -0.757 , 0.588; H 0. , 0.757 , 0.587')
    f1 = cost_function(mol, pop_method, exponent)
    mol.set_geom_('O 0. 0. 0.; H 0. , -0.757 , 0.586; H 0. , 0.757 , 0.587')
    f2 = cost_function(mol, pop_method, exponent)
    g1 = (f1 - f2) / (0.002 / BOHR)
    assert abs(g0[1,2]-g1) < 1e-4

def test_pm_cost_nuc_grad(get_mol):
    _cost_nuc_grad(get_mol, 'mulliken', 2)

def test_ibo_cost_nuc_grad(get_mol):
    _cost_nuc_grad(get_mol, 'iao', 4)

