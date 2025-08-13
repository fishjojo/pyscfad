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

@pytest.fixture
def get_h2():
    mol = gto.Mole()
    mol.atom    = 'H 0 0 0; H 0 0 0.74'
    mol.basis   = '631g'
    mol.verbose = 0
    mol.incore_anyway = True
    mol.max_memory = 7000
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    yield mol

@pytest.fixture
def get_h2o():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = '631G'
    mol.verbose=0
    mol.max_memory = 7000
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    yield mol

@pytest.fixture
def get_h2o_plus():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = 'sto3g'
    mol.spin = 1
    mol.charge = 1
    mol.verbose=0
    mol.max_memory = 7000
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    yield mol

@pytest.fixture
def get_n2():
    mol = gto.Mole()
    mol.atom = 'N 0 0 0; N 0 0 1.09'
    mol.basis = '631g'
    mol.verbose=0
    mol.max_memory = 7000
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    yield mol
