# Copyright 2021-2026 The PySCFAD Authors
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
from pyscfad import config_update
from pyscfad import gto, scf, cc
from pyscfad.cc import ccsd_t, ccsd_t_slow

@pytest.fixture
def get_mol_h2o():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. -0.757 0.587; H 0. 0.757 0.587'
    mol.basis = 'sto3g'
    mol.verbose = 0
    mol.incore_anyway = True
    mol.max_memory = 7000
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

def _energy_t(kernel):
    def energy(mol):
        mf = scf.RHF(mol)
        mf.kernel()
        mycc = cc.RCCSD(mf)
        _, t1, t2 = mycc.kernel()
        eris = mycc.ao2mo(mycc.mo_coeff)
        return kernel(mycc, eris, t1, t2)
    return energy

def test_ccsd_t_nuc_grad(get_mol_h2o):
    """The C kernel and its VJP must agree with the pure python
    implementation, which is differentiated by JAX directly.

    ``rccsd`` stores the ``ovvv`` integrals unpacked, which used to make
    the VJP return a cotangent of the wrong shape.
    """
    mol = get_mol_h2o
    with config_update('pyscfad_scf_implicit_diff', True), \
         config_update('pyscfad_ccsd_implicit_diff', True):
        e1, g1 = jax.value_and_grad(_energy_t(ccsd_t.kernel))(mol)
        e0, g0 = jax.value_and_grad(_energy_t(ccsd_t_slow.kernel))(mol)
    assert abs(e1 - e0) < 1e-10
    assert abs(g1.coords - g0.coords).max() < 1e-9
