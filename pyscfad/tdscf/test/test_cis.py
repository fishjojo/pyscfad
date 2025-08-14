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
import pyscf
from pyscfad import gto, scf, tdscf
from pyscfad import config

@pytest.fixture
def get_mol():
    config.update('pyscfad_scf_implicit_diff', True)

    mol = gto.Mole()
    mol.atom = 'H 0 0 0; F 0 0 1.09'
    mol.basis = '6-31G*'
    mol.verbose=0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    yield mol

    config.reset()

def test_nuc_grad(get_mol):
    mol = get_mol
    def energy(mol):
        mf = scf.RHF(mol)
        e_hf = mf.kernel()
        mytd = tdscf.rhf.CIS(mf)
        mytd.nstates = 3
        e = mytd.kernel()[0]
        return e[2] + e_hf
    g = jax.grad(energy)(mol).coords
    g0 = numpy.asarray([[0., 0.,  0.1319969988],
                        [0., 0., -0.1319969988]])
    assert abs(g-g0).max() < 1e-6
