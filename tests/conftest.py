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

from functools import partial
import pytest
from pyscfad import gto
from .util import make_mol

@pytest.fixture
def mol_H2():
    atom = "H 0 0 0; H 0 0 0.74"
    yield partial(make_mol, atom=atom)

@pytest.fixture
def mol_H2O():
    atom = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
    yield partial(make_mol, atom=atom)

@pytest.fixture
def mol_N2():
    atom = "N 0 0 0; N 0 0 1.09"
    yield partial(make_mol, atom=atom)

