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

"""The jittable DIIS of :mod:`pyscfad.lib.diis_lite` against the
buffer-based :class:`pyscfad.lib.diis.DIIS`.
"""
import numpy
import pytest
import jax

from pyscfad import numpy as np
from pyscfad.lib.diis import DIIS as DIISRef
from pyscfad.lib.diis_lite import DIISLite, solve_coefficients

SPACE = 6
NDIM = 12


@pytest.fixture(scope='module')
def linear_map():
    """A slowly converging contraction ``x -> A x + b`` and its fixed point.

    The spectral radius is put close to one, so the bare iteration crawls and
    the extrapolation has something to do -- as in a real coupled-cluster
    solve. A strongly contractive map converges on its own before the error
    vectors span a useful subspace.
    """
    rng = numpy.random.default_rng(1)
    a = rng.standard_normal((NDIM, NDIM))
    a *= .95 / max(abs(numpy.linalg.eigvals(a)))
    b = rng.standard_normal(NDIM)
    xstar = numpy.linalg.solve(numpy.eye(NDIM) - a, b)
    return a, b, xstar


@pytest.mark.parametrize('nvec', [1, 2, 3, SPACE])
def test_solve_coefficients(nvec):
    """The masked, full-size subspace equations reproduce the dense solve of
    the occupied block, and give no weight to the unused slots."""
    rng = numpy.random.default_rng(nvec)
    err = rng.standard_normal((nvec, NDIM))
    s = err @ err.T

    ovlp = numpy.zeros((SPACE, SPACE))
    ovlp[:nvec,:nvec] = s
    mask = numpy.arange(SPACE) < nvec
    c = numpy.asarray(solve_coefficients(np.asarray(ovlp), np.asarray(mask)))

    h = numpy.zeros((nvec+1, nvec+1))
    h[0,1:] = h[1:,0] = 1
    h[1:,1:] = s
    g = numpy.zeros(nvec+1)
    g[0] = 1
    c0 = numpy.linalg.solve(h, g)[1:]

    assert abs(c[:nvec] - c0).max() < 1e-12
    assert abs(c[nvec:]).max(initial=0.) == 0.
    assert abs(c.sum() - 1.) < 1e-12


def test_diis_matches_reference(linear_map):
    """Given the same vectors and error vectors, the two implementations
    extrapolate identically."""
    a, b, _ = linear_map
    dref = DIISRef()
    dref.incore = True
    dref.space = SPACE
    dnew = DIISLite(np.zeros(NDIM), space=SPACE)

    x_ref = numpy.zeros(NDIM)
    x_new = numpy.zeros(NDIM)
    for _ in range(2 * SPACE):  # long enough to wrap the circular buffer
        y = a @ x_ref + b
        x_ref = numpy.asarray(dref.update(y, y - x_ref))
        y = a @ x_new + b
        x_new = numpy.asarray(dnew.update(np.asarray(y), np.asarray(y - x_new)))
        assert abs(x_ref - x_new).max() < 1e-10


def test_diis_in_while_loop(linear_map):
    """The mixer is a pytree, so it rides in the loop carry, and it converges
    faster than the bare iteration."""
    a, b, xstar = linear_map
    ncycle = 30

    def solve(b, mix):
        def body(value):
            cycle, x, diis = value
            y = a @ x[0] + b
            if mix:
                x, diis = diis.update((y,)), diis
            else:
                x = (y,)
            return cycle+1, x, diis
        init = (0, (np.zeros(NDIM),), DIISLite((np.zeros(NDIM),), space=SPACE))
        return jax.lax.while_loop(lambda v: v[0] < ncycle, body, init)[1][0]

    err_plain = abs(numpy.asarray(jax.jit(solve, static_argnums=1)(b, False)) - xstar).max()
    err_diis = abs(numpy.asarray(jax.jit(solve, static_argnums=1)(b, True)) - xstar).max()
    assert err_diis < err_plain / 100


def test_diis_mixed_pytree():
    """Leaves of different shape and rank are extrapolated together."""
    rng = numpy.random.default_rng(0)
    shapes = [(3,), (2, 4), (2, 2, 2)]
    target = [np.asarray(rng.standard_normal(s)) for s in shapes]

    diis = DIISLite(tuple(np.zeros(s) for s in shapes), space=SPACE)
    x = tuple(np.zeros(s) for s in shapes)
    for _ in range(SPACE + 2):
        # a contraction towards target, so the fixed point is target itself
        y = tuple(xi + .5 * (ti - xi) for xi, ti in zip(x, target))
        x = diis.update(y)
    for xi, ti in zip(x, target):
        assert abs(numpy.asarray(xi) - numpy.asarray(ti)).max() < 1e-8
