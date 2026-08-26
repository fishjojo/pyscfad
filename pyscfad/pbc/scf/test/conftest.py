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

import numpy
import pytest
from pyscfad.pbc import gto

@pytest.fixture
def cell_H2():
    """Smallest cell that can be pushed through a pbc mean-field build."""
    cell = gto.Cell()
    cell.atom = 'H 0. 0. 0.; H 0. 0. 1.'
    cell.a = numpy.eye(3) * 2.
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.exp_to_discard = 0.1
    cell.mesh = [5,]*3
    cell.rcut = 2.
    cell.verbose = 0
    cell.build()
    return cell
