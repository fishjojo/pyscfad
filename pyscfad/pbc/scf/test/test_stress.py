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

import jax
from pyscfad import numpy as np
from pyscfad.pbc import gto, scf

def test_stree():
    def khf_energy(strain):
        cell = gto.Cell()
        cell.atom = """
            H 0. 0. 0.
            H 1. 1. 1.
        """
        cell.a = np.eye(3) * 2.
        cell.basis = "gth-szv"
        cell.pseudo = "gth-pade"
        cell.exp_to_discard = 0.1
        cell.mesh = [5]*3
        cell.rcut = 2.
        cell.build(trace_lattice_vectors=True)

        cell.abc += np.einsum("ab,nb->na", strain, cell.lattice_vectors())
        cell.coords += np.einsum("xy,ny->nx", strain, cell.atom_coords())

        mf = scf.KRHF(cell, kpts=np.zeros((1,3)), exxdiv=None)
        ehf = mf.kernel()
        return ehf

    strain = np.zeros((3,3))
    stress = jax.jacrev(khf_energy)(strain)

    stress0 = np.array(
        [[0.14468224, 0.01405739, 0.01405739],
         [0.01405739, 0.14468224, 0.01405739],
         [0.01405739, 0.01405739, 0.14468224]]
    )
    assert abs(stress - stress0).max() < 1e-6
