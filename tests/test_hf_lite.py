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

"""Tests for the lightweight, fully-jittable HF (:mod:`~pyscfad.scf.hf_lite`)."""
import pytest
import numpy
import jax
import pyscf
from pyscfad import numpy as np
from pyscfad.gto import MoleLite
from pyscfad.scf import hf_lite

BASIS = "sto3g"


@pytest.fixture
def water():
    symbols = ("O", "H", "H")
    coords = numpy.array([[0.0, 0.0, 0.0],
                          [0.0, -1.43, 1.1],
                          [0.0, 1.43, 1.1]])
    return symbols, coords


@pytest.fixture
def hydroxyl():
    symbols = ("O", "H")
    coords = numpy.array([[0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.8]])
    return symbols, coords


def _pyscf_mol(symbols, coords, spin=0):
    atom = list(zip(symbols, numpy.asarray(coords).tolist()))
    return pyscf.M(atom=atom, basis=BASIS, unit="Bohr", spin=spin, verbose=0)


def _e_rhf(coords, symbols, diis="diis"):
    mol = MoleLite(symbols=symbols, coords=coords, basis=BASIS,
                   trace_coords=True, verbose=0)
    mf = hf_lite.RHF(mol)
    mf.max_cycle = 80
    mf.conv_tol = 1e-11
    mf.diis = diis
    return mf.kernel()


def _e_uhf(coords, symbols, spin=1, diis="diis"):
    mol = MoleLite(symbols=symbols, coords=coords, basis=BASIS, spin=spin,
                   trace_coords=True, verbose=0)
    mf = hf_lite.UHF(mol)
    mf.max_cycle = 120
    mf.conv_tol = 1e-11
    mf.diis = diis
    return mf.kernel()


def test_rhf_energy(water):
    symbols, coords = water
    ref = _pyscf_mol(symbols, coords).RHF().run(conv_tol=1e-12).e_tot
    e = _e_rhf(coords, symbols)
    e_jit = jax.jit(lambda c: _e_rhf(c, symbols))(coords)
    assert abs(float(e) - ref) < 1e-9
    assert abs(float(e_jit) - float(e)) < 1e-10


@pytest.mark.parametrize("diis", [None, "anderson", "diis"])
def test_rhf_diis_modes(water, diis):
    symbols, coords = water
    ref = _pyscf_mol(symbols, coords).RHF().run(conv_tol=1e-12).e_tot
    assert abs(float(_e_rhf(coords, symbols, diis)) - ref) < 1e-9


def test_rhf_nuc_grad(water):
    symbols, coords = water
    pmf = _pyscf_mol(symbols, coords).RHF().run(conv_tol=1e-12)
    g_ref = pmf.nuc_grad_method().kernel()
    g = jax.jit(jax.grad(lambda c: _e_rhf(c, symbols)))(coords)
    assert abs(numpy.asarray(g) - g_ref).max() < 1e-5


def test_uhf_energy(hydroxyl):
    symbols, coords = hydroxyl
    ref = _pyscf_mol(symbols, coords, spin=1).UHF().run(conv_tol=1e-12).e_tot
    e = _e_uhf(coords, symbols)
    assert abs(float(e) - ref) < 1e-9


def test_uhf_nuc_grad(hydroxyl):
    symbols, coords = hydroxyl
    pmf = _pyscf_mol(symbols, coords, spin=1).UHF().run(conv_tol=1e-12)
    g_ref = pmf.nuc_grad_method().kernel()
    g = jax.jit(jax.grad(lambda c: _e_uhf(c, symbols)))(coords)
    assert abs(numpy.asarray(g) - g_ref).max() < 1e-5


def test_uhf_equals_rhf_closed_shell(water):
    symbols, coords = water
    e_rhf = _e_rhf(coords, symbols)
    e_uhf = _e_uhf(coords, symbols, spin=0)
    assert abs(float(e_rhf) - float(e_uhf)) < 1e-9
