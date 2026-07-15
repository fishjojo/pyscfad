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
from pyscfad.gto.moleintor_lite import _pair_index_matrix


@pytest.fixture
def h2o():
    # H2O at a bent geometry, coordinates in Bohr.
    sym = ("O", "H", "H")
    coords = np.asarray([[0.0, 0.0, 0.213],
                         [0.0, 1.43, -0.85],
                         [0.0, -1.43, -0.85]])
    return sym, coords


@pytest.fixture
def h2():
    sym = ("H", "H")
    coords = np.asarray([[0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.4]])
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


def _unpack_int2e(packed, aosym, nao):
    """Expand the leading packed int2e axes back to the full s1 layout."""
    pair_idx = _pair_index_matrix(nao)
    if aosym == "s2ij":
        return packed[pair_idx]
    if aosym == "s2kl":
        return packed[:, :, pair_idx]
    if aosym == "s4":
        return packed[pair_idx][:, :, pair_idx]
    if aosym == "s8":
        npair = nao * (nao + 1) // 2
        return _unpack_int2e(packed[_pair_index_matrix(npair)], "s4", nao)
    raise ValueError(f"Unknown aosym {aosym}")


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


def test_int1e_nuc_second_derivative_jit_matches_eager(h2o):
    # Exercises the vmapped rinv-at-nucleus jvp under jit at second order.
    sym, coords = h2o
    hess_fn = jax.jacfwd(jax.jacfwd(_intor_fn(sym, "int1e_nuc", hermi=1)))
    eager = numpy.asarray(hess_fn(coords))
    jitted = numpy.asarray(jax.jit(hess_fn)(coords))
    assert abs(eager - jitted).max() < 1e-10


def test_int1e_nuc_fwd_over_rev(h2o):
    # jacfwd(grad(...)) is the common hessian idiom; the batched per-nucleus
    # rinv jvp must stay composable under forward-over-reverse (a lax.scan
    # implementation broke exactly this composition).
    sym, coords = h2o
    fn = _intor_fn(sym, "int1e_nuc", hermi=1)
    hess = numpy.asarray(jax.jacfwd(jax.grad(lambda c: fn(c).sum()))(coords))
    ref = numpy.asarray(jax.jacfwd(jax.jacfwd(fn))(coords)).sum(axis=(0, 1))
    assert abs(hess - ref).max() < 1e-10


@pytest.mark.parametrize("name,kwargs", [
    ("int1e_ovlp", {"hermi": 1}),
    ("int1e_kin", {"hermi": 1}),
    ("int1e_nuc", {"hermi": 1}),
    ("int2e", {}),
])
def test_intor_second_derivative_vs_findiff(h2o, name, kwargs):
    sym, coords = h2o
    jac_fn = jax.jit(jax.jacfwd(_intor_fn(sym, name, **kwargs)))
    hess = numpy.asarray(jax.jacfwd(jac_fn)(coords))
    ref = _finite_diff(jac_fn, coords)
    assert abs(hess - ref).max() < 1e-6


@pytest.mark.parametrize("aosym", ["s2ij", "s2kl", "s4", "s8"])
def test_int2e_packed_aosym_matches_s1(h2o, aosym):
    sym, coords = h2o
    fn_s1 = _intor_fn(sym, "int2e")
    fn_packed = _intor_fn(sym, "int2e", aosym=aosym)
    nao = fn_s1(coords).shape[0]

    prim_s1 = numpy.asarray(fn_s1(coords))
    prim = numpy.asarray(_unpack_int2e(fn_packed(coords), aosym, nao))
    assert abs(prim - prim_s1).max() < 1e-12

    jac_s1 = numpy.asarray(jax.jacfwd(fn_s1)(coords))
    jac = numpy.asarray(_unpack_int2e(jax.jacfwd(fn_packed)(coords), aosym, nao))
    assert abs(jac - jac_s1).max() < 1e-12


def test_int2e_s4_second_derivative_matches_s1(h2o):
    sym, coords = h2o
    hess_s1 = numpy.asarray(jax.jacfwd(jax.jacfwd(_intor_fn(sym, "int2e")))(coords))
    fn_s4 = _intor_fn(sym, "int2e", aosym="s4")
    hess_s4 = jax.jacfwd(jax.jacfwd(fn_s4))(coords)
    nao = hess_s1.shape[0]
    assert abs(numpy.asarray(_unpack_int2e(hess_s4, "s4", nao)) - hess_s1).max() < 1e-12

    hess_s4_jit = jax.jit(jax.jacfwd(jax.jacfwd(fn_s4)))(coords)
    assert abs(numpy.asarray(hess_s4_jit) - numpy.asarray(hess_s4)).max() < 1e-10


def test_fourth_order_derivative_supported(h2):
    # libcintad ships derivative integrals up to total order 4; seed the chain
    # at dr20 so two jacfwd levels exercise the dr30/dr40 names cheaply.
    sym, coords = h2
    fn = _intor_fn(sym, "int1e_ovlp_dr20")
    d4 = numpy.asarray(jax.jacfwd(jax.jacfwd(fn))(coords))
    assert abs(d4).max() > 0
    # Partial derivatives commute: swapping differentiation steps is a no-op.
    assert abs(d4 - d4.transpose(0, 1, 2, 5, 6, 3, 4)).max() < 1e-10


def test_fourth_order_nested_jacfwd_high_cost(h2):
    # Full end-to-end 4-fold nesting starting from the plain overlap.
    sym, coords = h2
    fn = _intor_fn(sym, "int1e_ovlp", hermi=1)
    d4 = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(fn))))(coords)
    d4 = numpy.asarray(d4)
    assert abs(d4).max() > 0
    assert abs(d4 - d4.transpose(0, 1, 4, 5, 2, 3, 6, 7, 8, 9)).max() < 1e-10


def test_fifth_order_derivative_raises(h2):
    # Differentiating a total-order-4 integral needs order-5 kernels that
    # libcintad does not provide; the jvp must fail loudly.
    sym, coords = h2
    for name in ("int1e_ovlp_dr40", "int2e_dr4000"):
        fn = _intor_fn(sym, name)
        with pytest.raises(NotImplementedError):
            jax.jacfwd(fn)(coords)
