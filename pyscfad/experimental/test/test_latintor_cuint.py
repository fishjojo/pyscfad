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

import jax
from pyscf.data.nist import BOHR
from pyscfad import numpy as np
from pyscfad.pbc.gto import CellLite
from pyscfad.pbc.tools import nimgs_to_lattice_Ls
from pyscfad.experimental.moleintor_cuint import cuint_create_plan
from pyscfad.lib import hermi_triu

def func_norm(coords, numbers, a, basis, kmesh, cuint_plan=None):
    cell = CellLite(numbers=numbers, coords=coords, a=a, rcut=None,
                    basis=basis, precision=1e-6, trace_coords=True)
    kpts = cell.make_kpts(kmesh)
    Ls = nimgs_to_lattice_Ls(cell)
    expkL = np.exp(1j*np.dot(kpts, Ls.T))

    s1e_lat = cell.lattice_intor("int1e_ovlp", hermi=1, Ls=Ls, cuint_plan=cuint_plan)

    if cuint_plan is not None:
        h1 = np.einsum("kl,lpq->kpq", expkL, s1e_lat)
        h1 = h1 + h1.transpose(0,2,1).conj()
    else:
        h1 = np.einsum("kl,lpq->kpq", expkL, s1e_lat)
        h1 = hermi_triu(h1)

    res = jax.vmap(lambda s: np.sqrt(np.sum(s*s.conj()).real))(h1)
    return res

def test_latovlp():
    numbers = [14,14]
    coords = np.array(
        [
            [0.00000,  0.00000,  0.00000],
            [1.3467560987, 1.3467560987, 1.3467560987]
        ]
    ) / BOHR

    a = np.array([[0.0, 2.6935121974, 2.6935121974],
                  [2.6935121974, 0.0, 2.6935121974],
                  [2.6935121974, 2.6935121974, 0.0]]) / BOHR

    basis = "ccpvtz"
    cell = CellLite(numbers=numbers, coords=coords, a=a, rcut=None,
                    basis=basis, precision=1e-6, trace_coords=True)

    cuint_plan = cuint_create_plan(cell)
    kmesh = [3,2,2]
    s0 = func_norm(coords, numbers, a, basis, kmesh)
    s1 = func_norm(coords, numbers, a, basis, kmesh, cuint_plan)
    assert abs(s1-s0).max() < 1e-9

    g0 = jax.jacrev(func_norm)(coords, numbers, a, basis, kmesh)
    g1 = jax.jacrev(func_norm)(coords, numbers, a, basis, kmesh, cuint_plan)
    assert abs(g1-g0).max() < 1e-9
