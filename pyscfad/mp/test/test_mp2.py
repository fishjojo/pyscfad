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
import numpy
import jax
from pyscfad import gto, scf, mp
from pyscfad import config

@pytest.fixture
def get_mol():
    config.update('pyscfad_scf_implicit_diff', True)
    #config.update('pyscfad_moleintor_opt', True)

    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = '6-31G*'
    mol.verbose = 0
    mol.max_memory = 8000
    mol.incore_anyway = True
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    yield mol

    config.reset()

def mp2(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    mymp = mp.MP2(mf)
    mymp.kernel()
    return mymp.e_tot

def test_nuc_grad(get_mol):
    mol = get_mol
    g = jax.grad(mp2)(mol).coords
    # analytic gradient
    g0 = numpy.array(
         [[0, 0,            0.0132353292],
          [0, 0.0088696799,-0.0066176646],
          [0,-0.0088696799,-0.0066176646]])
    assert abs(g-g0).max() < 1e-6

def test_nuc_grad_n2():
    mol = gto.Mole()
    mol.atom = 'N 0 0 0; N 0 0 1.1'
    mol.basis = '6-31G*'
    mol.verbose = 0
    mol.max_memory = 8000
    mol.incore_anyway = True
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    g = jax.grad(mp2)(mol).coords
    # analytic gradient
    g0 = numpy.array(
        [[0, 0,  0.0821997739],
         [0, 0, -0.0821997739]])
    assert abs(g-g0).max() < 1e-6

def test_df_nuc_grad(get_mol):
    mol = get_mol
    def mp2(mol):
        mf = scf.RHF(mol).density_fit()
        e_hf = mf.kernel()
        mymp = mp.dfmp2.MP2(mf)
        mymp.kernel()
        return mymp.e_tot

    g = jax.grad(mp2)(mol).coords
    # finite difference
    g0 = numpy.array(
         [[0, 0,            0.0132337328],
          [0, 0.0088741480,-0.0066168447],
          [0,-0.0088741480,-0.0066168447]])
    assert abs(g-g0).max() < 1e-6
