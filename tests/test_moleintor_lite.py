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

"""Tests for the jittable/batchable :mod:`~pyscfad.gto.moleintor_lite` jvp rules
(``int1e_nuc`` and ``int2e``) migrated from the legacy ``_moleintor_jvp`` branch.
"""
import pytest
import numpy
import jax

from pyscfad import numpy as np
from pyscfad.gto.mole_lite import MoleLite
from pyscfad.gto import Mole as LegacyMole

SYMBOLS = ("O", "H", "H")
# atomic coordinates in Bohr (a single source of truth for all builds)
COORDS = numpy.array([[0.,  0.,       0.],
                      [0., -1.430564, 1.109132],
                      [0.,  1.430564, 1.109132]])
BASIS = "sto-3g"
CASES = [("int1e_nuc", 1), ("int1e_nuc", 0), ("int2e", 0)]

def _atom():
    return [(s, tuple(c)) for s, c in zip(SYMBOLS, COORDS.tolist())]

def _lite_intor(coords, name, hermi):
    m = MoleLite(symbols=SYMBOLS, coords=coords, basis=BASIS, trace_coords=True)
    return m.intor(name, hermi=hermi)

def _legacy_intor(coords, name, hermi):
    m = LegacyMole()
    m.atom = _atom()
    m.basis = BASIS
    m.unit = "Bohr"
    m.verbose = 0
    m.build(trace_coords=True)
    m.coords = coords
    return m.intor(name, hermi=hermi)

@pytest.fixture
def mol_lite():
    return MoleLite(symbols=SYMBOLS, coords=COORDS, basis=BASIS, verbose=0, trace_coords=True)

@pytest.mark.parametrize("name,hermi", CASES)
def test_forward_matches_pyscf(mol_lite, name, hermi):
    import pyscf
    ref = pyscf.M(atom=_atom(), basis=BASIS, unit="Bohr", verbose=0).intor(name, hermi=hermi)
    got = numpy.asarray(mol_lite.intor(name, hermi=hermi))
    assert abs(got - ref).max() < 1e-10

@pytest.mark.parametrize("name,hermi", CASES)
def test_jvp_vs_finite_difference(name, hermi):
    f = lambda c: _lite_intor(c, name, hermi)
    numpy.random.seed(0)
    v = numpy.asarray(numpy.random.randn(*COORDS.shape))
    _, tangent = jax.jvp(f, (np.asarray(COORDS),), (np.asarray(v),))
    eps = 1e-5
    fd = (numpy.asarray(f(COORDS + eps * v))
          - numpy.asarray(f(COORDS - eps * v))) / (2 * eps)
    assert abs(numpy.asarray(tangent) - fd).max() < 1e-6

@pytest.mark.parametrize("name,hermi", CASES)
def test_jvp_matches_legacy(name, hermi):
    """The migrated jvp must reproduce the legacy analytic jvp."""
    numpy.random.seed(1)
    v = np.asarray(numpy.random.randn(*COORDS.shape))
    _, t_lite = jax.jvp(lambda c: _lite_intor(c, name, hermi), (np.asarray(COORDS),), (v,))
    _, t_legacy = jax.jvp(lambda c: _legacy_intor(c, name, hermi), (np.asarray(COORDS),), (v,))
    assert abs(numpy.asarray(t_lite) - numpy.asarray(t_legacy)).max() < 1e-9

@pytest.mark.parametrize("name,hermi", CASES)
def test_jit_grad(name, hermi):
    scalar = lambda c: np.sum(_lite_intor(c, name, hermi) ** 2)
    g = jax.grad(scalar)(np.asarray(COORDS))
    gj = jax.jit(jax.grad(scalar))(np.asarray(COORDS))
    assert abs(numpy.asarray(g) - numpy.asarray(gj)).max() < 1e-9

def test_vmap_grad_int2e():
    scalar = lambda c: np.sum(_lite_intor(c, "int2e", 0) ** 2)
    coords = np.asarray(COORDS)
    batch = np.stack([coords, coords * 1.02, coords * 0.98])
    gb = jax.jit(jax.vmap(jax.grad(scalar)))(batch)
    g0 = jax.grad(scalar)(coords)
    assert gb.shape == (3,) + COORDS.shape
    assert abs(numpy.asarray(gb[0]) - numpy.asarray(g0)).max() < 1e-9
