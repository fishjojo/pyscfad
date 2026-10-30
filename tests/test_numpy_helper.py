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

""":func:`pyscfad.lib.unpack_tril` and :func:`pyscfad.lib.pack_tril` against
PySCF's, under jit and vmap, and in forward and reverse mode.
"""
import numpy
import pytest
import jax
from pyscf import lib as pyscf_lib
from pyscfad import numpy as np
from pyscfad import lib, config_update

N = 7
FILLS = (lib.SYMMETRIC, lib.HERMITIAN, lib.ANTIHERMI, lib.PLAIN)


@pytest.fixture(scope='module')
def rng():
    return numpy.random.default_rng(2)


def _random(rng, shape, dtype):
    x = rng.standard_normal(shape)
    if dtype == numpy.complex128:
        x = x + 1j * rng.standard_normal(shape)
    return x


DIAG = numpy.arange(N) * (numpy.arange(N) + 3) // 2 # packed positions of a[i,i]


def _ref_unpack(tril, filltriu):
    out = pyscf_lib.unpack_tril(tril.reshape(-1, tril.shape[-1]), filltriu=filltriu)
    if filltriu == lib.PLAIN: # pyscf leaves the upper triangle unset
        out = numpy.tril(out)
    # the diagonal is taken from tril; pyscf conjugates (HERMITIAN) or
    # negates and conjugates (ANTIHERMI) it with the upper triangle
    out[:, numpy.arange(N), numpy.arange(N)] = tril.reshape(-1, tril.shape[-1])[:, DIAG]
    return out.reshape(tril.shape[:-1] + (N, N))


@pytest.mark.parametrize('dtype', [numpy.float64, numpy.complex128])
@pytest.mark.parametrize('filltriu', FILLS)
@pytest.mark.parametrize('batch', [(), (3,), (2, 3)])
def test_unpack_tril(rng, filltriu, dtype, batch):
    tril = _random(rng, batch + (N*(N+1)//2,), dtype)
    ref = _ref_unpack(tril, filltriu)
    out = lib.unpack_tril(np.asarray(tril), filltriu)
    assert abs(out - ref).max() < 1e-14
    jitted = jax.jit(lib.unpack_tril, static_argnums=1)(np.asarray(tril), filltriu)
    assert abs(jitted - ref).max() < 1e-14
    if batch:
        vmapped = jax.vmap(lib.unpack_tril, (0, None))(np.asarray(tril), filltriu)
        assert abs(vmapped - ref).max() < 1e-14
    if len(batch) == 1:
        # packed along the leading axis: (npair, count) -> (n, n, count)
        tril0 = numpy.ascontiguousarray(tril.T)
        ref0 = pyscf_lib.unpack_tril(tril0, filltriu=filltriu, axis=0)
        out = lib.unpack_tril(np.asarray(tril0), filltriu, axis=0)
        assert out.shape == ref0.shape
        if filltriu == lib.PLAIN: # pyscf leaves the upper triangle unset
            lower = numpy.tril(numpy.ones((N, N), dtype=bool))[:,:,None]
            ref0 = numpy.where(lower, ref0, 0)
        ref0[numpy.arange(N), numpy.arange(N)] = tril0[DIAG]
        assert abs(out - ref0).max() < 1e-14


@pytest.mark.parametrize('filltriu', FILLS)
def test_unpack_tril_derivatives(rng, filltriu):
    tril = np.asarray(rng.standard_normal((3, N*(N+1)//2)))
    tdot = np.asarray(rng.standard_normal(tril.shape))
    g = np.asarray(rng.standard_normal((3, N, N)))
    f = lambda t: lib.unpack_tril(t, filltriu)
    # the map is linear: its JVP is the map, and the VJP its adjoint
    assert abs(jax.jvp(f, (tril,), (tdot,))[1] - f(tdot)).max() < 1e-14
    tbar = jax.vjp(f, tril)[1](g)[0]
    assert abs(np.vdot(f(tdot), g) - np.vdot(tdot, tbar)) < 1e-12


@pytest.mark.parametrize('dtype', [numpy.float64, numpy.complex128])
@pytest.mark.parametrize('batch', [(), (3,)])
def test_pack_tril(rng, dtype, batch):
    a = _random(rng, batch + (N, N), dtype)
    ref = pyscf_lib.pack_tril(a)
    assert abs(lib.pack_tril(np.asarray(a)) - ref).max() < 1e-14
    assert abs(jax.jit(lib.pack_tril)(np.asarray(a)) - ref).max() < 1e-14
    if batch:
        # the leading two axes, packed along axis 0
        a0 = numpy.ascontiguousarray(numpy.moveaxis(a, 0, -1))
        ref0 = pyscf_lib.pack_tril(a0, axis=0)
        out = lib.pack_tril(np.asarray(a0), axis=0)
        assert out.shape == ref0.shape
        assert abs(out - ref0).max() < 1e-14
    sym = np.asarray(a + numpy.swapaxes(a, -1, -2))
    assert abs(lib.unpack_tril(lib.pack_tril(sym), lib.SYMMETRIC) - sym).max() < 1e-14
    if batch:
        sym0 = np.moveaxis(sym, 0, -1)
        out = lib.unpack_tril(lib.pack_tril(sym0, axis=0), lib.SYMMETRIC, axis=0)
        assert abs(out - sym0).max() < 1e-14


def test_pack_tril_derivatives(rng):
    a = np.asarray(rng.standard_normal((3, N, N)))
    adot = np.asarray(rng.standard_normal(a.shape))
    g = np.asarray(rng.standard_normal((3, N*(N+1)//2)))
    assert abs(jax.jvp(lib.pack_tril, (a,), (adot,))[1] - lib.pack_tril(adot)).max() < 1e-14
    abar = jax.vjp(lib.pack_tril, a)[1](g)[0]
    assert abs(np.vdot(lib.pack_tril(adot), g) - np.vdot(adot, abar)) < 1e-12


def test_invalid_shapes():
    """The gathers clip their indices, so invalid sizes are rejected instead
    of being clamped silently."""
    with pytest.raises(ValueError):
        lib.unpack_tril(np.ones(5))
    with pytest.raises(ValueError):
        lib.pack_tril(np.ones((3, 4)))
    with pytest.raises(ValueError):
        lib.pack_tril(np.ones((3, 4, 2)), axis=0)


def test_moleintor_opt(rng):
    """Without a host round trip, both trace under jit with
    ``pyscfad_moleintor_opt`` as well."""
    a = np.asarray(rng.standard_normal((N, N)))
    with config_update('pyscfad_moleintor_opt', True):
        tril = jax.jit(lib.pack_tril)(a)
        out = jax.jit(lib.unpack_tril, static_argnums=1)(tril, lib.SYMMETRIC)
    assert abs(out - (np.tril(a) + np.tril(a, -1).T)).max() < 1e-14
