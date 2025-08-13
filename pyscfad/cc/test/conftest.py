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
from pyscfad import gto
from pyscfad import config

@pytest.fixture
def get_mol():
    config.update('pyscfad_scf_implicit_diff', True)
    config.update('pyscfad_ccsd_implicit_diff', True)

    mol = gto.Mole()
    mol.atom = 'H 0. 0. 0.; F 0. 0. 1.1'
    mol.basis = '631g'
    mol.verbose = 0
    mol.incore_anyway = True
    mol.max_memory = 7000
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    yield mol

    config.reset()

@pytest.fixture
def get_opt_mol():
    config.update("pyscfad_moleintor_opt", True)
    config.update('pyscfad_scf_implicit_diff', True)
    config.update('pyscfad_ccsd_implicit_diff', True)

    mol = gto.Mole()
    mol.atom = 'H 0. 0. 0.; F 0. 0. 1.1'
    mol.basis = '631g'
    mol.verbose = 0
    mol.incore_anyway = True
    mol.max_memory = 7000
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    yield mol

    config.reset()
