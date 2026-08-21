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

"""First derivatives of one-electron integrals with respect to the basis-set
parameters of a :class:`~pyscfad.gto.MoleLite`, i.e. w.r.t. ``MoleLite.basis``.

``MoleLite.basis`` is a ``{symbol: {l: [block]}}`` tree whose leaves are the
``(nprim, 1 + nctr)`` parameter blocks of one shell: column 0 holds the
exponents and columns ``1:`` the contraction coefficients.
The derivative therefore splits into the ``exp`` and ``ctr_coeff`` parts as in
:class:`~pyscfad.gto.Mole`.

Note that the derivative is taken w.r.t. the raw basis parameters,
so it runs through the shell normalization when building ``_env``.
The normalization makes ``ctr_coeff`` depend on the ``exp``.
"""
import contextlib

import numpy
import pytest
import jax

from pyscfad import numpy as np
from pyscfad.gto import MoleLite

SYMBOLS = ("H1", "H2")
BASIS = {
    "H1": [
        [0, [3.42, 0.15, 0.06], [0.62, 0.55, 0.35]],
        [1, [0.9, 1.0]],
        [2, [0.1, 1.0]],
    ],
    "H2": "sto3g",
}
COORDS = numpy.array([[0.0, 0.0, 0.0], [0.1, 0.0, 1.4]])
ORIGIN = numpy.array([0.3, -0.2, 0.6])

INTORS = [
    "int1e_ovlp",
    "int1e_kin",
    "int1e_nuc",
    "int1e_ovlp_dr10",
    "int1e_ovlp_dr01",
    "int1e_kin_dr10",
    "int1e_kin_dr01",
    "int1e_nuc_dr10",
    "int1e_nuc_dr01",
]


@pytest.fixture(scope="module")
def basis():
    return MoleLite(symbols=SYMBOLS, coords=np.asarray(COORDS),
                    basis=BASIS).basis


@pytest.fixture(scope="module")
def ao_loc(basis):
    return MoleLite(symbols=SYMBOLS, coords=np.asarray(COORDS),
                    basis=basis).ao_loc


def intor_fn(intor, hermi=0, cart=False, shls_slice=None, origin=None,
             aosym="s1"):
    """``basis -> integral`` for a fixed geometry. ``origin`` places the
    gauge origin of the operator at :data:`ORIGIN`, "common" for
    ``int1e_r``-type and "rinv" for ``int1e_rinv``-type integrals.
    """
    def fn(basis):
        mol = MoleLite(symbols=SYMBOLS, coords=np.asarray(COORDS),
                       basis=basis, cart=cart)
        if origin == "common":
            ctx = mol.with_common_origin(ORIGIN)
        elif origin == "rinv":
            ctx = mol.with_rinv_origin(ORIGIN)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            return mol.intor(intor, hermi=hermi, shls_slice=shls_slice,
                             aosym=aosym)
    return fn


def four_point_fd(fn, basis, disp=1e-4):
    """4-point finite differences of ``fn(basis)`` w.r.t. every basis
    parameter, as a list aligned with ``jax.tree.leaves(basis)``; each entry
    has shape ``fn(basis).shape + leaf.shape``.
    """
    leaves, treedef = jax.tree.flatten(basis)
    grad_fd = []
    for i, leaf in enumerate(leaves):
        leaf = numpy.asarray(leaf, dtype=float)
        g = None
        for idx in numpy.ndindex(leaf.shape):
            def at(d):
                leaf1 = leaf.copy()
                leaf1[idx] += d
                leaves1 = list(leaves)
                leaves1[i] = np.asarray(leaf1)
                return numpy.asarray(fn(jax.tree.unflatten(treedef, leaves1)))

            sp, sm = at(disp), at(-disp)
            sp2, sm2 = at(2 * disp), at(-2 * disp)
            if g is None:
                g = numpy.zeros(sp.shape + leaf.shape)
            g[(...,) + idx] = (8.0 * (sp - sm) - (sp2 - sm2)) / (12.0 * disp)
        grad_fd.append(g)
    return grad_fd


def assert_cs_exp_close(jac, grad_fd, tol):
    """Compare the coefficient and exponent parts of every shell block."""
    for leaf, g_fd in zip(jax.tree.leaves(jac), grad_fd):
        leaf = numpy.asarray(leaf)
        assert leaf.shape == g_fd.shape
        # column 0 of a block are the exponents, columns 1: the coefficients
        assert abs(leaf[..., 1:] - g_fd[..., 1:]).max() < tol  # cs
        assert abs(leaf[..., 0] - g_fd[..., 0]).max() < tol    # exp


@pytest.mark.parametrize("intor", INTORS)
def test_cs_exp(basis, intor):
    hermi = 0 if "_dr" in intor else 1
    fn = intor_fn(intor, hermi=hermi)

    jac_fwd = jax.jacfwd(fn)(basis)
    grad_fd = four_point_fd(fn, basis)
    assert_cs_exp_close(jac_fwd, grad_fd, 1e-7)

    jac_rev = jax.jacrev(fn)(basis)
    for l_fwd, l_rev in zip(jax.tree.leaves(jac_fwd), jax.tree.leaves(jac_rev)):
        assert abs(numpy.asarray(l_fwd) - numpy.asarray(l_rev)).max() < 1e-12


@pytest.mark.parametrize("intor,origin", [("int1e_r", "common"),
                                          ("int1e_rinv", "rinv")])
def test_cs_exp_origin(basis, intor, origin):
    """Operators carrying a gauge origin. The origin is a plain constant in
    ``_env``, so only the basis parameters contribute to the tangent.
    """
    fn = intor_fn(intor, hermi=1, origin=origin)

    jac_fwd = jax.jacfwd(fn)(basis)
    grad_fd = four_point_fd(fn, basis)
    assert_cs_exp_close(jac_fwd, grad_fd, 1e-7)

    jac_rev = jax.jacrev(fn)(basis)
    for l_fwd, l_rev in zip(jax.tree.leaves(jac_fwd), jax.tree.leaves(jac_rev)):
        assert abs(numpy.asarray(l_fwd) - numpy.asarray(l_rev)).max() < 1e-12


@pytest.mark.parametrize("intor", ["int1e_ovlp", "int1e_kin", "int1e_nuc"])
def test_cs_exp_cart(basis, intor):
    """Cartesian AOs skip the cartesian-to-spherical transformation of the
    exponent derivative.
    """
    fn = intor_fn(intor, hermi=1, cart=True)

    jac = jax.jacfwd(fn)(basis)
    grad_fd = four_point_fd(fn, basis)
    assert_cs_exp_close(jac, grad_fd, 1e-7)


SLICES = [
    ((0, 2, 1, 3), 0),  # rectangular off-diagonal block
    ((1, 3, 1, 3), 1),  # diagonal block, hermitian
    ((2, 3, 0, 1), 0),  # a single shell on either side
    ((0, 1, 0, 3), 0),  # part of the rows, all of the columns
]


@pytest.mark.parametrize("shls_slice,hermi", SLICES)
def test_cs_exp_shls_slice(basis, ao_loc, shls_slice, hermi):
    """Basis derivatives of a partial shell block. Bra and ket select
    different shell ranges, so the two cross blocks differ.
    """
    fn = intor_fn("int1e_ovlp", hermi=hermi, shls_slice=shls_slice)

    jac = jax.jacfwd(fn)(basis)
    assert_cs_exp_close(jac, four_point_fd(fn, basis), 1e-7)

    # the sliced derivative is the matching block of the full one
    i0, i1, j0, j1 = shls_slice
    jac_full = jax.jacfwd(intor_fn("int1e_ovlp", hermi=hermi))(basis)
    for leaf, leaf_full in zip(jax.tree.leaves(jac), jax.tree.leaves(jac_full)):
        # the two trailing axes of a leaf are the (nprim, 1 + nctr) block
        ref = numpy.asarray(leaf_full)[..., ao_loc[i0]:ao_loc[i1],
                                       ao_loc[j0]:ao_loc[j1], :, :]
        leaf = numpy.asarray(leaf)
        assert leaf.shape == ref.shape
        assert abs(leaf - ref).max() < 1e-12


def test_cs_exp_jit(basis):
    """The basis derivative is jittable and jit-invariant."""
    fn = intor_fn("int1e_ovlp", hermi=1)

    jac = jax.jacrev(fn)(basis)
    jac_jit = jax.jit(jax.jacrev(fn))(basis)
    for leaf, leaf_jit in zip(jax.tree.leaves(jac), jax.tree.leaves(jac_jit)):
        assert abs(numpy.asarray(leaf) - numpy.asarray(leaf_jit)).max() < 1e-12


def test_cs_exp_mixed_coords_basis(basis):
    """d/d(basis) of the coordinate-gradient norm (coords inner, basis outer).
    """
    coords = np.asarray(COORDS)

    def gnorm(basis):
        def inner(coords_):
            mol = MoleLite(symbols=SYMBOLS, coords=coords_, basis=basis)
            return np.linalg.norm(mol.intor("int1e_ovlp", hermi=1))
        return np.linalg.norm(jax.grad(inner)(coords))

    grad = jax.grad(gnorm)(basis)
    assert_cs_exp_close(grad, four_point_fd(gnorm, basis), 1e-6)


@pytest.mark.parametrize("hermi", [1, 0])
def test_cs_exp_mixed_basis_coords(basis, hermi):
    """d/d(coords) of the basis-gradient norm (basis inner, coords outer).
    """
    coords = np.asarray(COORDS)

    def gnorm(coords_):
        def inner(basis_):
            mol = MoleLite(symbols=SYMBOLS, coords=coords_, basis=basis_)
            return np.linalg.norm(mol.intor("int1e_ovlp", hermi=hermi))
        grad = jax.grad(inner)(basis)
        return np.sqrt(sum(np.sum(leaf ** 2) for leaf in jax.tree.leaves(grad)))

    grad = numpy.asarray(jax.grad(gnorm)(coords))
    grad_fd = four_point_fd(gnorm, coords)[0]
    assert grad.shape == grad_fd.shape
    assert abs(grad - grad_fd).max() < 1e-6


@pytest.mark.parametrize("aosym", ["s4", "s8"])
def test_cs_exp_int2e(basis, aosym):
    """Basis derivatives of the packed two-electron integrals. Every one of
    the four indices carries the derivative; the permutation symmetry of
    ``(ij|kl)`` maps three of the four terms onto the first.
    """
    fn = intor_fn("int2e", aosym=aosym)

    jac = jax.jacfwd(fn)(basis)
    assert_cs_exp_close(jac, four_point_fd(fn, basis), 1e-7)

    # reverse mode, checked on a scalar contraction of the same jacobian
    out = numpy.asarray(fn(basis))
    grad = jax.grad(lambda b: np.sum(fn(b) ** 2))(basis)
    for leaf, leaf_fwd in zip(jax.tree.leaves(grad), jax.tree.leaves(jac)):
        ref = 2. * numpy.tensordot(out, numpy.asarray(leaf_fwd), axes=out.ndim)
        assert abs(numpy.asarray(leaf) - ref).max() < 1e-10


def test_cs_exp_int2e_cart(basis):
    """Cartesian AOs skip the cartesian-to-spherical transformation of the
    three spectator indices of the exponent derivative.
    """
    fn = intor_fn("int2e", aosym="s4", cart=True)

    jac = jax.jacfwd(fn)(basis)
    assert_cs_exp_close(jac, four_point_fd(fn, basis), 1e-7)


def test_cs_exp_int2e_shls_slice(basis):
    """A block whose bra and ket pair span different shells: the ket-side
    terms are their own cross integrals rather than a transpose.
    """
    fn = intor_fn("int2e", aosym="s4", shls_slice=(0, 3, 0, 3, 3, 4, 3, 4))

    jac = jax.jacfwd(fn)(basis)
    assert_cs_exp_close(jac, four_point_fd(fn, basis), 1e-7)


def test_cs_exp_int2e_mixed_unsupported(basis):
    """Mixed coordinate and basis derivatives of ``int2e``.

    Either nesting reaches an integral whose tangent is not available: with
    the coordinates inside, the basis tangent of a differentiated integral
    would need the permutation symmetry that the derivative has broken;
    with the basis inside, the cross integrals of a packed pair are not
    themselves packed the way the coordinate tangent expects.
    """
    coords = np.asarray(COORDS)

    def coords_inner(basis_):
        def inner(coords_):
            mol = MoleLite(symbols=SYMBOLS, coords=coords_, basis=basis_)
            return np.linalg.norm(mol.intor("int2e", aosym="s8"))
        return np.linalg.norm(jax.grad(inner)(coords))

    def basis_inner(coords_):
        def inner(basis_):
            mol = MoleLite(symbols=SYMBOLS, coords=coords_, basis=basis_)
            return np.linalg.norm(mol.intor("int2e", aosym="s8"))
        grad = jax.grad(inner)(basis)
        return np.sqrt(sum(np.sum(leaf ** 2) for leaf in jax.tree.leaves(grad)))

    with pytest.raises(NotImplementedError):
        jax.grad(coords_inner)(basis)
    with pytest.raises(NotImplementedError):
        jax.grad(basis_inner)(coords)


# int2e carries derivatives with aosym='s4' and 's8' only
@pytest.mark.parametrize("intor,aosym", [("int2e", "s1"), ("int2e", "s2ij")])
def test_unsupported_intor(basis, intor, aosym):
    with pytest.raises(NotImplementedError):
        jax.jacfwd(intor_fn(intor, aosym=aosym))(basis)
