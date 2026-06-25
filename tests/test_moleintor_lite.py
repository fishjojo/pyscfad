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

"""Tests for the jittable integral jvps in :mod:`pyscfad.gto.moleintor_lite`."""
import numpy
import pytest
import jax
from pyscfad import numpy as np
from pyscfad.gto.mole_lite import MoleLite


@pytest.fixture
def h2o():
    # H2O at a bent geometry, coordinates in Bohr.
    sym = ("O", "H", "H")
    coords = np.asarray([[0.0, 0.0, 0.213],
                         [0.0, 1.43, -0.85],
                         [0.0, -1.43, -0.85]])
    return sym, coords


def _intor_fn(sym, name, **kwargs):
    def fn(coords):
        mol = MoleLite(symbols=sym, coords=coords, basis="sto3g",
                       trace_coords=True, verbose=0)
        return mol.intor(name, **kwargs)
    return fn


def _finite_diff(fn, coords, disp=1e-5):
    base = numpy.asarray(fn(coords))
    grad = numpy.zeros(base.shape + coords.shape)
    for ia in range(coords.shape[0]):
        for x in range(3):
            plus = numpy.asarray(fn(coords.at[ia, x].add(disp)))
            minus = numpy.asarray(fn(coords.at[ia, x].add(-disp)))
            grad[..., ia, x] = (plus - minus) / (2 * disp)
    return grad


@pytest.mark.parametrize("name,kwargs", [
    ("int1e_ovlp", {"hermi": 1}),
    ("int1e_kin", {"hermi": 1}),
    ("int1e_nuc", {"hermi": 1}),
    ("int2e", {}),
])
def test_intor_jvp_vs_findiff(h2o, name, kwargs):
    sym, coords = h2o
    fn = _intor_fn(sym, name, **kwargs)
    jac = numpy.asarray(jax.jacfwd(fn)(coords))
    ref = _finite_diff(fn, coords)
    assert abs(jac - ref).max() < 1e-6


def test_int2e_jvp_reverse_matches_forward(h2o):
    sym, coords = h2o
    fn = _intor_fn(sym, "int2e")
    jf = numpy.asarray(jax.jacfwd(fn)(coords))
    jr = numpy.asarray(jax.jacrev(fn)(coords))
    assert abs(jf - jr).max() < 1e-9


def test_int1e_nuc_jvp_reverse_matches_forward(h2o):
    sym, coords = h2o
    fn = _intor_fn(sym, "int1e_nuc", hermi=1)
    jf = numpy.asarray(jax.jacfwd(fn)(coords))
    jr = numpy.asarray(jax.jacrev(fn)(coords))
    assert abs(jf - jr).max() < 1e-9


def test_intor_jvp_jit_matches_eager(h2o):
    sym, coords = h2o
    for name, kwargs in (("int1e_nuc", {"hermi": 1}), ("int2e", {})):
        fn = _intor_fn(sym, name, **kwargs)
        eager = numpy.asarray(jax.jacfwd(fn)(coords))
        jitted = numpy.asarray(jax.jit(jax.jacfwd(fn))(coords))
        assert abs(eager - jitted).max() < 1e-10


def test_higher_order_derivative_raises(h2o):
    sym, coords = h2o
    for name, kwargs in (("int1e_nuc", {"hermi": 1}), ("int2e", {})):
        fn = _intor_fn(sym, name, **kwargs)
        with pytest.raises(NotImplementedError):
            jax.jacfwd(jax.jacfwd(fn))(coords)
