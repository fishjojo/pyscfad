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

"""Tests for the jittable RHF/UHF mean-field in :mod:`pyscfad.scf.hf_lite`."""
import numpy
import pytest
import jax
import pyscf
from pyscf import scf as pyscf_scf

from pyscfad import numpy as np
from pyscfad.gto.mole_lite import MoleLite
from pyscfad.scf import hf_lite

BASIS = "sto3g"
# H2O, coordinates in Bohr.
SYM = ("O", "H", "H")
COORDS = np.asarray([[0.0, 0.0, 0.213],
                     [0.0, 1.43, -0.85],
                     [0.0, -1.43, -0.85]])


def _pyscf_mf(charge, spin, unrestricted, init_guess="hcore"):
    atom = [[s, tuple(x)] for s, x in zip(SYM, numpy.asarray(COORDS).tolist())]
    mol = pyscf.M(atom=atom, basis=BASIS, unit="AU",
                  charge=charge, spin=spin, verbose=0)
    mf = (pyscf_scf.UHF if unrestricted else pyscf_scf.RHF)(mol)
    mf.init_guess = init_guess
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def _energy_fn(cls, charge=0, spin=0, diis="diis"):
    def energy(coords):
        mol = MoleLite(symbols=SYM, coords=coords, basis=BASIS,
                       charge=charge, spin=spin, trace_coords=True, verbose=0)
        mf = cls(mol)
        mf.diis = diis
        mf.conv_tol = 1e-11
        return mf.kernel()
    return energy


def test_rhf_energy():
    e = float(_energy_fn(hf_lite.RHF)(COORDS))
    # closed-shell RHF is robust: matches the PySCF (minao) ground state
    e0 = _pyscf_mf(0, 0, False, init_guess="minao").e_tot
    assert abs(e - e0) < 1e-9


def test_rhf_nuc_grad():
    g = numpy.asarray(jax.grad(_energy_fn(hf_lite.RHF))(COORDS))
    g0 = _pyscf_mf(0, 0, False, init_guess="minao").nuc_grad_method().kernel()
    assert abs(g - g0).max() < 1e-6


def test_uhf_energy():
    # compare at the matching (hcore-guess) UHF solution
    e = float(_energy_fn(hf_lite.UHF, charge=1, spin=1)(COORDS))
    e0 = _pyscf_mf(1, 1, True, init_guess="hcore").e_tot
    assert abs(e - e0) < 1e-8


def test_uhf_nuc_grad():
    g = numpy.asarray(jax.grad(_energy_fn(hf_lite.UHF, charge=1, spin=1))(COORDS))
    g0 = _pyscf_mf(1, 1, True, init_guess="hcore").nuc_grad_method().kernel()
    assert abs(g - g0).max() < 1e-6


def test_uhf_closed_shell_equals_rhf():
    e_uhf = float(_energy_fn(hf_lite.UHF, charge=0, spin=0)(COORDS))
    e_rhf = float(_energy_fn(hf_lite.RHF, charge=0, spin=0)(COORDS))
    assert abs(e_uhf - e_rhf) < 1e-9


@pytest.mark.parametrize("diis", ["diis", "anderson", None])
def test_rhf_diis_variants_agree(diis):
    e = float(_energy_fn(hf_lite.RHF, diis=diis)(COORDS))
    e0 = _pyscf_mf(0, 0, False, init_guess="minao").e_tot
    assert abs(e - e0) < 1e-9


def test_rhf_jit_value_and_grad():
    fn = _energy_fn(hf_lite.RHF)
    e_j, g_j = jax.jit(jax.value_and_grad(fn))(COORDS)
    e_e, g_e = jax.value_and_grad(fn)(COORDS)
    assert abs(float(e_j) - float(e_e)) < 1e-10
    assert abs(numpy.asarray(g_j) - numpy.asarray(g_e)).max() < 1e-10


def test_uhf_jit_value_and_grad():
    fn = _energy_fn(hf_lite.UHF, charge=1, spin=1)
    e_j, g_j = jax.jit(jax.value_and_grad(fn))(COORDS)
    e_e, g_e = jax.value_and_grad(fn)(COORDS)
    assert abs(float(e_j) - float(e_e)) < 1e-10
    assert abs(numpy.asarray(g_j) - numpy.asarray(g_e)).max() < 1e-10
