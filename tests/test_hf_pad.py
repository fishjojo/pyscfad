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

"""Batched HF over padded molecules (:mod:`~pyscfad.ml.scf.hf_pad`).

A single ``jax.jit(jax.vmap(jax.value_and_grad(...)))`` is mapped over a batch of
chemically different, zero-padded molecules; energies and nuclear gradients are
checked per system against PySCF, and the gradient on padding (ghost) atoms must
vanish.
"""
import pytest
import numpy
import jax
import pyscf
from pyscfad import numpy as np
from pyscfad.ml.gto import MolePad, make_basis_array
from pyscfad.ml.scf import hf_pad

BASIS_NAME = "sto-3g"
MAX_Z = 8


@pytest.fixture(scope="module")
def basis():
    return make_basis_array(BASIS_NAME, MAX_Z)


@pytest.fixture
def rhf_batch():
    # H2O (3 atoms, padded with one ghost) and NH3 (4 atoms); both closed shell.
    numbers = numpy.array([[8, 1, 1, 0],
                           [7, 1, 1, 1]], dtype=numpy.int32)
    coords = numpy.array([
        [[0.0, 0.0, 0.0], [0.0, -1.43, 1.1], [0.0, 1.43, 1.1], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.9, 0.5], [1.6, -0.95, 0.5], [-1.6, -0.95, 0.5]],
    ])
    return numbers, coords


@pytest.fixture
def uhf_batch():
    # two OH doublets at different bond lengths
    numbers = numpy.array([[8, 1], [8, 1]], dtype=numpy.int32)
    coords = numpy.array([
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 2.1]],
    ])
    return numbers, coords


def _pyscf_ref(numbers, coords, spin=0, restricted=True):
    real = numbers > 0
    atom = [[int(z), tuple(map(float, c))]
            for z, c in zip(numbers[real], numpy.asarray(coords)[real])]
    mol = pyscf.M(atom=atom, basis=BASIS_NAME, unit="Bohr", spin=spin, verbose=0)
    mf = (mol.RHF() if restricted else mol.UHF()).run(conv_tol=1e-12)
    return mf.e_tot, mf.nuc_grad_method().kernel()


def test_rhf_pad_batched(basis, rhf_batch):
    numbers, coords = rhf_batch

    def energy(numbers, coords):
        mol = MolePad(numbers, coords, basis=basis, trace_coords=True, verbose=0)
        mf = hf_pad.RHF(mol)
        mf.max_cycle = 80
        mf.conv_tol = 1e-10
        mf.diis = "diis"
        return mf.kernel()

    vg = jax.jit(jax.vmap(jax.value_and_grad(energy, argnums=1), (0, 0)))
    energies, grads = vg(numbers, coords)
    energies = numpy.asarray(energies)
    grads = numpy.asarray(grads)

    for i in range(len(numbers)):
        e_ref, _ = _pyscf_ref(numbers[i], coords[i])
        assert abs(energies[i] - e_ref) < 1e-8
        # gradient on the padding (ghost) atoms must vanish
        pad = numbers[i] == 0
        if pad.any():
            assert abs(grads[i][pad]).max() < 1e-9


def test_uhf_pad_batched(basis, uhf_batch):
    numbers, coords = uhf_batch

    def energy(numbers, coords):
        mol = MolePad(numbers, coords, basis=basis, spin=1,
                      trace_coords=True, verbose=0)
        mf = hf_pad.UHF(mol)
        mf.max_cycle = 120
        mf.conv_tol = 1e-10
        mf.diis = "diis"
        return mf.kernel()

    vg = jax.jit(jax.vmap(jax.value_and_grad(energy, argnums=1), (0, 0)))
    energies, grads = vg(numbers, coords)
    energies = numpy.asarray(energies)
    grads = numpy.asarray(grads)

    for i in range(len(numbers)):
        e_ref, g_ref = _pyscf_ref(numbers[i], coords[i], spin=1, restricted=False)
        assert abs(energies[i] - e_ref) < 1e-8
        assert abs(grads[i] - g_ref).max() < 1e-5
