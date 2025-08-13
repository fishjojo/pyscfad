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

@pytest.fixture(autouse=True)
def clear_cache():
    yield
    jax.clear_caches()

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
    mol.basis = '631g'
    mol.max_memory = 7000
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    yield mol

@pytest.fixture
def get_mol_p():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74005'  # in Angstrom
    mol.basis = '631g'
    mol.build()
    yield mol

@pytest.fixture
def get_mol_m():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.73995'  # in Angstrom
    mol.basis = '631g'
    mol.build()
    yield mol
