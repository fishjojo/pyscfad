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

"""Tests for the jittable :mod:`~pyscfad.scf.hf_lite` RHF/UHF mean-field."""
import pytest
import numpy
import jax

from pyscfad import numpy as np
from pyscfad.gto.mole_lite import MoleLite
from pyscfad.scf import hf_lite

BASIS = "sto-3g"
# H2O (closed shell) and OH (doublet), coordinates in Bohr
H2O = ("O 0 0 0; H 0 -1.430564 1.109132; H 0 1.430564 1.109132", 0)
OH = ("O 0 0 0; H 0 0 1.834345", 1)

def _mol(atom, spin):
    import pyscf
    p = pyscf.M(atom=atom, basis=BASIS, unit="Bohr", spin=spin, verbose=0)
    return MoleLite.from_pyscf(p, trace_coords=True), p

def _energy_fn(symbols, basis, spin, cls, diis=None):
    def efn(coords):
        m = MoleLite(symbols=symbols, coords=coords, basis=basis, spin=spin,
                     trace_coords=True)
        mf = cls(m)
        mf.max_cycle = 80
        mf.conv_tol = 1e-11
        mf.diis = diis
        return mf.kernel()
    return efn

# ----------------------------- RHF -----------------------------------------

def test_rhf_energy():
    ml, p = _mol(*H2O)
    e = float(hf_lite.RHF(ml).kernel())
    assert abs(e - p.RHF().run().e_tot) < 1e-9

def test_rhf_gradient():
    ml, p = _mol(*H2O)
    efn = _energy_fn(ml.symbols, ml.basis, 0, hf_lite.RHF)
    g = numpy.asarray(jax.grad(efn)(ml.coords))
    gref = p.RHF().run().nuc_grad_method().kernel()
    assert abs(g - gref).max() < 1e-5

def test_rhf_jit():
    ml, _ = _mol(*H2O)
    efn = _energy_fn(ml.symbols, ml.basis, 0, hf_lite.RHF)
    e = efn(ml.coords)
    assert abs(float(jax.jit(efn)(ml.coords)) - float(e)) < 1e-9
    g = jax.grad(efn)(ml.coords)
    gj = jax.jit(jax.grad(efn))(ml.coords)
    assert abs(numpy.asarray(g) - numpy.asarray(gj)).max() < 1e-9

# ----------------------------- UHF -----------------------------------------

def test_uhf_energy_open_shell():
    ml, p = _mol(*OH)
    e = float(hf_lite.UHF(ml).kernel())
    assert abs(e - p.UHF().run().e_tot) < 1e-9

def test_uhf_gradient_open_shell():
    ml, p = _mol(*OH)
    efn = _energy_fn(ml.symbols, ml.basis, 1, hf_lite.UHF)
    g = numpy.asarray(jax.grad(efn)(ml.coords))
    gref = p.UHF().run().nuc_grad_method().kernel()
    assert abs(g - gref).max() < 1e-5

def test_uhf_equals_rhf_closed_shell():
    ml, _ = _mol(*H2O)
    e_rhf = float(hf_lite.RHF(ml).kernel())
    e_uhf = float(hf_lite.UHF(ml).kernel())
    assert abs(e_rhf - e_uhf) < 1e-9

def test_uhf_jit():
    ml, _ = _mol(*OH)
    efn = _energy_fn(ml.symbols, ml.basis, 1, hf_lite.UHF)
    g = jax.grad(efn)(ml.coords)
    gj = jax.jit(jax.grad(efn))(ml.coords)
    assert abs(numpy.asarray(g) - numpy.asarray(gj)).max() < 1e-9

# ----------------------------- DIIS ----------------------------------------

@pytest.mark.parametrize("cls,atom,spin", [(hf_lite.RHF, H2O[0], H2O[1]),
                                           (hf_lite.UHF, OH[0], OH[1])])
@pytest.mark.parametrize("diis", [None, "anderson", "diis"])
def test_diis_variants_agree(cls, atom, spin, diis):
    ml, p = _mol(atom, spin)
    ref = (p.UHF() if cls is hf_lite.UHF else p.RHF()).run().e_tot
    efn = _energy_fn(ml.symbols, ml.basis, spin, cls, diis=diis)
    e = float(jax.jit(efn)(ml.coords))
    assert abs(e - ref) < 1e-9
