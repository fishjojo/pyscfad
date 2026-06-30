# Copyright 2025-2026 The PySCFAD Authors
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

"""Tests for batched (padded) RHF/UHF in :mod:`pyscfad.ml.scf.hf_pad`."""
from functools import partial
import numpy
import pytest
import jax
import pyscf
from pyscf import scf as pyscf_scf

from pyscfad import numpy as np
from pyscfad.ml.gto import MolePad, make_basis_array
from pyscfad.ml.scf import RHF as RHFPad, UHF as UHFPad

BASIS = "sto3g"


@pytest.fixture(scope="module")
def basis_array():
    return make_basis_array(BASIS, max_number=8)


def _pyscf(sym, coords, charge, spin, unrestricted, init_guess):
    atom = [[s, tuple(x)] for s, x in zip(sym, numpy.asarray(coords).tolist())]
    mol = pyscf.M(atom=atom, basis=BASIS, unit="AU",
                  charge=charge, spin=spin, verbose=0)
    mf = (pyscf_scf.UHF if unrestricted else pyscf_scf.RHF)(mol)
    mf.init_guess = init_guess
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf.e_tot, mf.nuc_grad_method().kernel()


def test_batched_rhf(basis_array):
    # H2O (3 atoms) and H2 (2 atoms + 1 padding atom).
    numbers = np.asarray([[8, 1, 1], [1, 1, 0]], dtype=int)
    coords = np.asarray([
        [[0.0, 0.0, 0.213], [0.0, 1.43, -0.85], [0.0, -1.43, -0.85]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 0.0, 0.0]],
    ])

    e0_h2o, g0_h2o = _pyscf(("O", "H", "H"), coords[0], 0, 0, False, "minao")
    e0_h2, g0_h2 = _pyscf(("H", "H"), coords[1][:2], 0, 0, False, "minao")

    def energy(numbers, coords):
        mol = MolePad(numbers, coords, basis=basis_array,
                      trace_coords=True, verbose=0)
        mf = RHFPad(mol)
        mf.diis = "diis"
        mf.conv_tol = 1e-11
        return mf.kernel()

    e, g = jax.jit(jax.vmap(jax.value_and_grad(energy, 1)))(numbers, coords)
    e = numpy.asarray(e)
    g = numpy.asarray(g)

    assert abs(e[0] - e0_h2o) < 1e-9
    assert abs(e[1] - e0_h2) < 1e-9
    assert abs(g[0, :3] - g0_h2o).max() < 1e-6
    assert abs(g[1, :2] - g0_h2).max() < 1e-6
    # padding atom must not contribute to the gradient
    assert abs(g[1, 2]).max() < 1e-10


def test_batched_uhf(basis_array):
    # OH (2 atoms + 1 padding) and NH2 (3 atoms), both doublets.
    numbers = np.asarray([[8, 1, 0], [7, 1, 1]], dtype=int)
    coords = np.asarray([
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.83], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.45, 1.0], [0.0, -1.45, 1.0]],
    ])

    # compare at the matching (hcore-guess) UHF solution
    e0_oh, g0_oh = _pyscf(("O", "H"), coords[0][:2], 0, 1, True, "hcore")
    e0_nh2, g0_nh2 = _pyscf(("N", "H", "H"), coords[1], 0, 1, True, "hcore")

    def energy(numbers, coords):
        mol = MolePad(numbers, coords, basis=basis_array, spin=1,
                      trace_coords=True, verbose=0)
        mf = UHFPad(mol)
        mf.diis = "diis"
        mf.conv_tol = 1e-11
        return mf.kernel()

    e, g = jax.jit(jax.vmap(jax.value_and_grad(energy, 1)))(numbers, coords)
    e = numpy.asarray(e)
    g = numpy.asarray(g)

    assert abs(e[0] - e0_oh) < 1e-8
    assert abs(e[1] - e0_nh2) < 1e-8
    assert abs(g[0, :2] - g0_oh).max() < 1e-6
    assert abs(g[1, :3] - g0_nh2).max() < 1e-6
    # padding atom must not contribute to the gradient
    assert abs(g[0, 2]).max() < 1e-10
