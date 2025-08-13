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

def test_nuc_hessian(get_mol):
    mol = get_mol
    def energy(mol):
        mf = scf.RHF(mol)
        mf.kernel()
        mycc = cc.RCCSD(mf)
        mycc.kernel()
        return mycc.e_tot
    h1 = jax.jacrev(jax.jacrev(energy))(mol).coords.coords
    h0 = numpy.array(
        [[[[ 4.20246014e-02, 0, 0],
           [-4.20246014e-02, 0, 0]],
          [[0,  4.20246014e-02, 0],
           [0, -4.20246014e-02, 0]],
          [[0, 0,  1.53246547e-01],
           [0, 0, -1.53246547e-01]]],
         [[[-4.20246015e-02, 0, 0],
           [ 4.20246015e-02, 0, 0]],
          [[0, -4.20246017e-02, 0],
           [0,  4.20246017e-02, 0]],
          [[0, 0, -1.53241777e-01],
           [0, 0,  1.53241780e-01]]]]
    )
    assert(abs(h1-h0).max() < 5e-5)
