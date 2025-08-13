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
import numpy
from pyscfad.pbc import dft

def test_nuc_grad(get_Si2):
    cell = get_Si2
    kpts = cell.make_kpts([2,1,1])

    def dft_energy(cell, kpts):
        mf = dft.KRKS(cell, kpts=kpts, exxdiv=None)
        mf.xc = 'pbe'
        e_tot = mf.kernel()
        return e_tot

    jac = jax.grad(dft_energy)(cell, kpts)
    g0 = numpy.asarray([[-0.0151487891,  0.0023280773,  0.0023280773],
                        [ 0.0151486150, -0.0023404710, -0.0023404710]])
    assert abs(jac.coords - g0).max() < 1e-5
