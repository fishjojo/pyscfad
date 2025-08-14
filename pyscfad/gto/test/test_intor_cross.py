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
from pyscfad import gto
from pyscfad.gto import mole

TEST_SET = ["int1e_ovlp", "int1e_kin", "int1e_rinv", "int2c2e"]

@pytest.fixture
def get_h2o():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = 'ccpvdz'
    mol.verbose=0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

def ints_cross_single(intor, mol):
    return mole.intor_cross(intor, mol, mol)

def ints_cross(intor, mol1, mol2):
    return mole.intor_cross(intor, mol1, mol2)

def ints(intor, mol):
    return mol.intor(intor)

def test_nuc_grad(get_h2o):
    mol = get_h2o
    for intor in TEST_SET:
        s0 = ints(intor, mol)
        s1 = ints_cross_single(intor, mol)
        s2 = ints_cross(intor, mol, mol)
        assert abs(s1 - s0).max() < 1e-9
        assert abs(s2 - s0).max() < 1e-9

        g0 = jax.jacrev(ints, 1)(intor, mol).coords
        g1 = jax.jacrev(ints_cross_single, 1)(intor, mol).coords
        jac2 = jax.jacrev(ints_cross, (1,2))(intor, mol, mol)
        g2 = jac2[0].coords + jac2[1].coords
        assert abs(g1 - g0).max() < 1e-9
        assert abs(g2 - g0).max() < 1e-9
