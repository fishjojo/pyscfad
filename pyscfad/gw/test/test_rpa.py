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
from pyscf import df as pyscf_df
from pyscfad import gto, dft, df
from pyscfad.gw import rpa
from pyscfad import config_update

@pytest.fixture
def get_h2o():
    with config_update('pyscfad_scf_implicit_diff', True):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            [8 , [0. , 0.     , 0.]],
            [1 , [0. , -0.7571 , 0.5861]],
            [1 , [0. , 0.7571 , 0.5861]]]
        mol.basis = 'def2-svp'
        mol.max_memory = 8000
        mol.incore_anyway = True
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        yield mol

def test_nuc_grad(get_h2o):
    mol = get_h2o
    auxbasis = pyscf_df.addons.make_auxbasis(mol, mp2fit=True)
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    with_df = df.DF(mol, auxmol=auxmol)

    def energy(mol, with_df):
        mf = dft.RKS(mol)
        mf.xc = 'pbe'
        mf.kernel()

        mymp = rpa.RPA(mf)
        mymp.with_df = with_df
        mymp.kernel()
        return mymp.e_tot

    jac = jax.grad(energy, (0,1))(mol, with_df)
    g1 = jac[0].coords + jac[1].mol.coords + jac[1].auxmol.coords
    g0 = numpy.array([[0.,  0.,              2.04900196e-02],
                      [0.,  7.11947436e-03, -1.02450070e-02],
                      [0., -7.11947447e-03, -1.02450070e-02]])
    assert(abs(g1-g0).max() < 1e-5)
