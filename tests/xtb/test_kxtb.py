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

import pytest
import jax
from pyscf.data.nist import BOHR
from pyscfad import numpy as np
from pyscfad.gto import MoleLite as Mole
from pyscfad.xtb import basis as xtb_basis
from pyscfad.xtb import GFN1XTB
from pyscfad.xtb.param import GFN1Param
from pyscfad.pbc.gto import CellLite as Cell
from pyscfad.xtb.kxtb import GFN1KXTB

@pytest.fixture
def setup():
    basis = xtb_basis.get_basis_filename()
    param = GFN1Param()
    yield basis, param

def test_gfn1_kxtb_energy_force(setup, H2O_GFN1_ref):
    basis, param = setup
    numbers, coords, *_ = H2O_GFN1_ref

    def mol_energy(coords, diis):
        mol = Mole(numbers=numbers, coords=coords, basis=basis, trace_coords=True)
        mf = GFN1XTB(mol, param=param)
        mf.diis = diis
        return mf.kernel()

    def cell_energy(coords, diis):
        cell = Cell(numbers=numbers, coords=coords, a=np.eye(3)*20., rcut=22.,
                    basis=basis, precision=1e-6, trace_coords=True)
        mf = GFN1KXTB(cell, param=param)
        mf.diis = diis
        return mf.kernel()

    for diis in ("anderson", "qbroyden"):
        e0, g0 = jax.value_and_grad(mol_energy)(coords, diis)
        e1, g1 = jax.value_and_grad(cell_energy)(coords, diis)

        assert abs(e1 - e0) < 1e-3
        assert abs(g1 - g0).max() < 1e-3

def test_gfn1_kxtb_energy_force_with_kpts_sample(setup):
    numbers = [14,14]
    coords = np.asarray([[0.0, 0.0, 0.0],
                         [1.3467560987, 1.3467560987, 1.3467560987]]) / BOHR
    a = np.asarray([[0.0, 2.6935121974, 2.6935121974],
                    [2.6935121974, 0.0, 2.6935121974],
                    [2.6935121974, 2.6935121974, 0.0]]) / BOHR
    basis, param = setup

    def cell_energy(coords):
        cell = Cell(numbers=numbers, coords=coords, a=a,
                    basis=basis, precision=1e-6, trace_coords=True)
        mf = GFN1KXTB(cell, param=param, kpts=cell.make_kpts([2,]*3))
        mf.diis = "anderson"
        return mf.kernel()

    e0 = -3.81064113210415
    g0 = np.array([[-0.00073464,]*3, [0.00073464,]*3])

    e1, g1 = jax.value_and_grad(cell_energy)(coords)
    assert abs(e1 - e0) < 1e-6
    assert abs(g1 - g0).max() < 1e-6

def test_gfn1_kxtb_smearing(setup):
    basis, param = setup
    numbers = [29, 29]
    coords = np.asarray(
        [[0.        , 0.        , 0.        ],
         [2.40522868, 2.40522868, 3.40150702],]
    )
    a = np.asarray(
        [[4.81045737, 0.        , 0.        ],
         [0.        , 4.81045737, 0.        ],
         [0.        , 0.        , 6.80301405],]
    )

    def cell_energy(coords):
        cell = Cell(numbers=numbers, coords=coords, a=a,
                    basis=basis, precision=1e-6, trace_coords=True)
        mf = GFN1KXTB(cell, param=param, kpts=cell.make_kpts([2,]*3))
        mf.sigma = 0.001
        mf.diis = "anderson"
        mf.diis_damp = .5
        mf.diis_space = 6
        return mf.kernel()

    e0 = -9.222835240828761
    g0 = np.zeros((2,3))

    e1, g1 = jax.value_and_grad(cell_energy)(coords)
    assert abs(e1 - e0) < 1e-6
    assert abs(g1 - g0).max() < 1e-6
