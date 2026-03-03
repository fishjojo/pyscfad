# Copyright 2026 The PySCFAD Authors
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

import numpy as np
import jax
from pyscf.data.nist import BOHR
from pyscf.pbc.gto import Cell
from pyscfad.pbc.gto import CellLite
from .test_pbc_intor import int1e_deriv1_r0, func_norm

INTOR = ["int1e_ovlp", "int1e_kin"]

def test_int1e():
    l = 2.6935121974
    a = np.array(
        [[0.0, l, l],
         [l, 0.0, l],
         [l, l, 0.0]]
    )
    atom = [["Si", [0,0,0]], ["Si", [.5*l,]*3]]

    for intor in INTOR:

        cell = Cell(atom=atom, basis="sto3g", a=a).build()
        kpts = cell.make_kpts([2,2,2])
        s1e_ref = np.asarray(cell.pbc_intor(intor, kpts=kpts))

        norm = np.asarray(func_norm(cell, intor, kpts))
        _g0 = np.asarray(int1e_deriv1_r0(cell, intor, kpts=kpts))
        g0 = []
        for i in range(len(kpts)):
            tmp = np.einsum("ijnx,ij->nx", _g0[i], s1e_ref[i].conj())
            tmp += tmp.conj()
            g0.append((tmp * 0.5 / norm[i]).real)
        g0 = np.asarray(g0)

        coords = np.array([[0,0,0],[.5*l,.5*l,.5*l]]) / BOHR
        cell = CellLite(numbers=[14,14], coords=coords, basis="sto3g", a=a/BOHR)
        kpts = cell.make_kpts([2,2,2])
        s1e = cell.pbc_intor(intor, kpts=kpts)

        def func(coords):
            cell = CellLite(numbers=[14,14], coords=coords, basis="sto3g", a=a/BOHR, trace_coords=True)
            kpts = cell.make_kpts([2,2,2])
            return func_norm(cell, intor, kpts=kpts)
        g1 = np.asarray(jax.jacrev(func)(coords))

        assert abs(s1e-s1e_ref).max() < 2e-8
        assert abs(g1-g0).max() < 2e-8
