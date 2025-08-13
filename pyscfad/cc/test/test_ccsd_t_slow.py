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
import jax
from pyscfad import scf, cc

def test_nuc_grad(get_mol):
    mol = get_mol
    def energy(mol):
        mf = scf.RHF(mol)
        mf.kernel()
        mycc = cc.RCCSD(mf)
        mycc.kernel()
        et = mycc.ccsd_t()
        return mycc.e_tot + et
    g1 = jax.grad(energy)(mol).coords
    g0 = numpy.array([[0., 0., -8.60709468e-02],
                      [0., 0.,  8.60709468e-02]])
    assert(abs(g1-g0).max() < 1e-6)
