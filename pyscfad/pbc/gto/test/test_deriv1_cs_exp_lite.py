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

"""First derivatives of the periodic one-electron integrals with respect to
the basis-set parameters of a :class:`~pyscfad.pbc.gto.CellLite`, i.e. w.r.t.
``CellLite.basis``.

Both entry points are covered: the per-image lattice integrals
(``lattice_intor``, :mod:`pyscfad.pbc.gto._latintor`) and the k-point
integrals (``pbc_intor``, :mod:`pyscfad.pbc.gto._pbcintor_lite`). The leaf
layout of ``basis`` and the exponent/coefficient column split are as in
:mod:`pyscfad.gto.test.test_deriv1_cs_exp_lite`.

With ``hermi = 1`` the lattice backend stores only the lower triangle (the
``s2`` fill) while the k-point matrices are completed to hermitian; the
tangent follows the primal in both cases, so finite differences of the
primal are the right oracle either way.
"""
import numpy
import pytest
import jax

from pyscf.data.nist import BOHR

from pyscfad import numpy as np
from pyscfad.pbc.gto import CellLite

# 2-atom silicon cell with a small (s, p) basis
A = numpy.array([[0.0, 2.6935, 2.6935],
                 [2.6935, 0.0, 2.6935],
                 [2.6935, 2.6935, 0.0]]) / BOHR
COORDS = numpy.array([[0.0, 0.0, 0.0], [1.3468] * 3]) / BOHR
NUMBERS = [14, 14]
BASIS = "gth-szv"
RCUT = 8.0

# shells: Si1 s, Si1 p, Si2 s, Si2 p, with ao_loc = [0, 1, 4, 5, 8];
# hermi = 1 is only defined for a diagonal shell block
SLICES = [((0, 2, 1, 4), 0), ((1, 4, 1, 4), 1)]


@pytest.fixture(scope="module")
def cell():
    return CellLite(numbers=NUMBERS, coords=COORDS, a=A, basis=BASIS,
                    rcut=RCUT, precision=1e-6)


@pytest.fixture(scope="module")
def nimgs(cell):
    """The image bounds of the reference cell.

    Passing them explicitly keeps ``CellLite.__init__`` from deriving them
    from ``_env`` (via the non-jittable ``get_lattice_Ls``), which is both
    what makes the construction jittable under a traced basis and a lot
    cheaper to repeat inside the finite-difference loops.
    """
    return tuple(int(x) for x in numpy.asarray(cell.nimgs))


@pytest.fixture(scope="module")
def Ls(cell):
    return numpy.asarray(cell.Ls, dtype=float).reshape(-1, 3)


@pytest.fixture(scope="module")
def kpts(cell):
    return numpy.asarray(cell.make_kpts([2, 1, 1]), dtype=float)


def lattice_fn(Ls, nimgs, hermi=0, cart=False, shls_slice=None):
    """``basis -> per-image lattice integrals`` for a fixed geometry."""
    def fn(basis):
        cell_ = CellLite(numbers=NUMBERS, coords=COORDS, a=A, basis=basis,
                         rcut=RCUT, nimgs=nimgs, precision=1e-6, cart=cart)
        return cell_.lattice_intor("int1e_ovlp", hermi=hermi, Ls=Ls,
                                   shls_slice=shls_slice)
    return fn


def pbc_fn(kpts, nimgs, hermi=0, shls_slice=None):
    """``basis -> k-point integrals`` for a fixed geometry."""
    def fn(basis):
        cell_ = CellLite(numbers=NUMBERS, coords=COORDS, a=A, basis=basis,
                         rcut=RCUT, nimgs=nimgs, precision=1e-6)
        return cell_.pbc_intor("int1e_ovlp", hermi=hermi, kpts=kpts,
                               shls_slice=shls_slice)
    return fn


def _sample_idx(shape):
    """A few positions of a ``(nprim, 1 + nctr)`` shell block, always
    including an exponent (column 0) and a contraction coefficient.
    """
    nprim, ncol = shape
    return sorted({(0, 0),
                   (nprim - 1, ncol - 1),
                   (nprim // 2, 1 + (ncol - 1) // 2)})


def four_point_fd(fn, basis, disp=1e-4):
    """4-point finite differences of ``fn(basis)`` at a few parameters of
    every shell block, as ``(leaf index, (prim, column), value)``.
    """
    leaves, treedef = jax.tree.flatten(basis)
    grad_fd = []
    for i, leaf in enumerate(leaves):
        leaf = numpy.asarray(leaf, dtype=float)
        for idx in _sample_idx(leaf.shape):
            def at(d):
                leaf1 = leaf.copy()
                leaf1[idx] += d
                leaves1 = list(leaves)
                leaves1[i] = np.asarray(leaf1)
                return numpy.asarray(fn(jax.tree.unflatten(treedef, leaves1)))

            g = (8.0 * (at(disp) - at(-disp))
                 - (at(2 * disp) - at(-2 * disp))) / (12.0 * disp)
            grad_fd.append((i, idx, g))
    return grad_fd


def assert_cs_exp_close(jac, grad_fd, tol=1e-7):
    """Compare the sampled coefficient and exponent derivatives."""
    leaves = jax.tree.leaves(jac)
    n_exp = n_cs = 0
    for i, idx, g_fd in grad_fd:
        g_ad = numpy.asarray(leaves[i])[(...,) + idx]
        assert g_ad.shape == g_fd.shape
        assert abs(g_ad - g_fd).max() < tol
        n_exp += idx[1] == 0
        n_cs += idx[1] > 0
    assert n_exp and n_cs, "neither part was actually checked"


@pytest.mark.parametrize("hermi", [0, 1])
def test_cs_exp_lattice(cell, Ls, nimgs, hermi):
    fn = lattice_fn(Ls, nimgs, hermi=hermi)

    jac_fwd = jax.jacfwd(fn)(cell.basis)
    assert_cs_exp_close(jac_fwd, four_point_fd(fn, cell.basis))

    jac_rev = jax.jacrev(fn)(cell.basis)
    for l_fwd, l_rev in zip(jax.tree.leaves(jac_fwd), jax.tree.leaves(jac_rev)):
        assert abs(numpy.asarray(l_fwd) - numpy.asarray(l_rev)).max() < 1e-12


def test_cs_exp_lattice_cart(cell, Ls, nimgs):
    """Cartesian AOs skip the cartesian-to-spherical transformation of the
    exponent derivative.
    """
    fn = lattice_fn(Ls, nimgs, hermi=1, cart=True)
    assert_cs_exp_close(jax.jacfwd(fn)(cell.basis), four_point_fd(fn, cell.basis))


@pytest.mark.parametrize("hermi", [0, 1])
def test_cs_exp_pbc(cell, kpts, nimgs, hermi):
    fn = pbc_fn(kpts, nimgs, hermi=hermi)

    jac_fwd = jax.jacfwd(fn)(cell.basis)
    assert_cs_exp_close(jac_fwd, four_point_fd(fn, cell.basis))

    # the k-point integrals are complex; use a real scalar loss for the
    # reverse-mode comparison
    def loss(basis):
        s = fn(basis)
        return np.sum(np.abs(s) ** 2)

    for g_rev, g_fwd in zip(jax.tree.leaves(jax.grad(loss)(cell.basis)),
                            jax.tree.leaves(jax.jacfwd(loss)(cell.basis))):
        assert abs(numpy.asarray(g_rev) - numpy.asarray(g_fwd)).max() < 1e-10


@pytest.mark.parametrize("shls_slice,hermi", SLICES)
def test_cs_exp_lattice_shls_slice(cell, Ls, nimgs, shls_slice, hermi):
    """Basis derivatives of a partial shell block; for ``hermi = 1`` the
    lower-triangle mask has to follow the sliced block.
    """
    fn = lattice_fn(Ls, nimgs, hermi=hermi, shls_slice=shls_slice)

    jac = jax.jacfwd(fn)(cell.basis)
    assert_cs_exp_close(jac, four_point_fd(fn, cell.basis))

    # the sliced derivative is the matching block of the full one
    ao_loc = numpy.asarray(cell.ao_loc)
    i0, i1, j0, j1 = shls_slice
    jac_full = jax.jacfwd(lattice_fn(Ls, nimgs, hermi=hermi))(cell.basis)
    for leaf, leaf_full in zip(jax.tree.leaves(jac), jax.tree.leaves(jac_full)):
        # the two trailing axes of a leaf are the (nprim, 1 + nctr) block
        ref = numpy.asarray(leaf_full)[..., ao_loc[i0]:ao_loc[i1],
                                       ao_loc[j0]:ao_loc[j1], :, :]
        leaf = numpy.asarray(leaf)
        assert leaf.shape == ref.shape
        assert abs(leaf - ref).max() < 1e-12


@pytest.mark.parametrize("shls_slice,hermi", SLICES)
def test_cs_exp_pbc_shls_slice(cell, kpts, nimgs, shls_slice, hermi):
    fn = pbc_fn(kpts, nimgs, hermi=hermi, shls_slice=shls_slice)

    jac = jax.jacfwd(fn)(cell.basis)
    assert_cs_exp_close(jac, four_point_fd(fn, cell.basis))

    ao_loc = numpy.asarray(cell.ao_loc)
    i0, i1, j0, j1 = shls_slice
    jac_full = jax.jacfwd(pbc_fn(kpts, nimgs, hermi=hermi))(cell.basis)
    for leaf, leaf_full in zip(jax.tree.leaves(jac), jax.tree.leaves(jac_full)):
        ref = numpy.asarray(leaf_full)[..., ao_loc[i0]:ao_loc[i1],
                                       ao_loc[j0]:ao_loc[j1], :, :]
        leaf = numpy.asarray(leaf)
        assert leaf.shape == ref.shape
        assert abs(leaf - ref).max() < 1e-12


def test_cs_exp_jit(cell, Ls, kpts, nimgs):
    """The basis derivative is jittable and jit-invariant."""
    for fn in (lattice_fn(Ls, nimgs, hermi=1), pbc_fn(kpts, nimgs, hermi=1)):
        def loss(basis):
            s = fn(basis)
            return np.sum(np.abs(s) ** 2)

        grad = jax.grad(loss)(cell.basis)
        grad_jit = jax.jit(jax.grad(loss))(cell.basis)
        for g, gj in zip(jax.tree.leaves(grad), jax.tree.leaves(grad_jit)):
            assert abs(numpy.asarray(g) - numpy.asarray(gj)).max() < 1e-12
