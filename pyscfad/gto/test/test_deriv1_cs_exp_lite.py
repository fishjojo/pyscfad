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
exponents and columns ``1:`` the contraction coefficients. The derivative
therefore splits into the ``exp`` and ``cs`` parts of every leaf, the
:class:`~pyscfad.gto.MoleLite` counterpart of ``jac.exp`` / ``jac.ctr_coeff``
in :mod:`~pyscfad.gto.test.test_deriv1_cs_exp`.

Note that the derivative is taken w.r.t. the basis parameters as given by the
user, so it runs through the shell normalization applied when building
``_env`` as well. The normalization makes the ``env`` contraction coefficients
depend on the exponents, so the exponent column of a leaf is not a pure
``exp`` derivative: a broken coefficient derivative shows up in both columns.
"""
import contextlib

import numpy
import pytest
import jax

from pyscfad import numpy as np
from pyscfad.gto import MoleLite

# H1 carries a general contraction (nprim = nctr = 2) and a p shell; the
# two atoms use different basis sets, so both shared and distinct env slots
# are exercised.
SYMBOLS = ("H1", "H2")
BASIS = {
    "H1": [
        [0, [3.42, 0.15, 0.06], [0.62, 0.55, 0.35]],
        [1, [0.9, 1.0]],
    ],
    "H2": "sto3g",
}
COORDS = numpy.array([[0.0, 0.0, 0.0], [0.1, 0.0, 1.4]])
# gauge origin of int1e_r / int1e_rinv, off both nuclei
ORIGIN = numpy.array([0.3, -0.2, 0.6])

INTORS = [
    "int1e_ovlp",
    "int1e_kin",
    "int1e_ovlp_dr10",
    "int1e_ovlp_dr01",
    "int1e_kin_dr10",
    "int1e_kin_dr01",
]


@pytest.fixture(scope="module")
def basis():
    """The formatted ``MoleLite.basis`` tree."""
    return MoleLite(symbols=SYMBOLS, coords=np.asarray(COORDS),
                    basis=BASIS).basis


@pytest.fixture(scope="module")
def ao_loc(basis):
    return MoleLite(symbols=SYMBOLS, coords=np.asarray(COORDS),
                    basis=basis).ao_loc


def intor_fn(intor, hermi=0, cart=False, shls_slice=None, origin=None):
    """``basis -> integral`` for a fixed geometry. ``origin`` places the
    gauge origin of the operator at :data:`ORIGIN`, "common" for
    ``int1e_r``-type and "rinv" for ``int1e_rinv``-type integrals.
    """
    def fn(basis):
        mol = MoleLite(symbols=SYMBOLS, coords=np.asarray(COORDS),
                       basis=basis, cart=cart, trace_basis=True)
        if origin == "common":
            ctx = mol.with_common_origin(ORIGIN)
        elif origin == "rinv":
            ctx = mol.with_rinv_origin(ORIGIN)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            return mol.intor(intor, hermi=hermi, shls_slice=shls_slice)
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


@pytest.mark.parametrize("intor", ["int1e_ovlp", "int1e_kin"])
def test_cs_exp_cart(basis, intor):
    """Cartesian AOs skip the cartesian-to-spherical transformation of the
    exponent derivative.
    """
    fn = intor_fn(intor, hermi=1, cart=True)

    jac = jax.jacfwd(fn)(basis)
    grad_fd = four_point_fd(fn, basis)
    assert_cs_exp_close(jac, grad_fd, 1e-7)


# shell 0 is the H1 s block (nctr = 2, AOs 0:2), shell 1 the H1 p block
# (AOs 2:5) and shell 2 the H2 s block (AO 5); hermi = 1 is only defined
# for a diagonal shell block.
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
    """d/d(basis) of the coordinate-gradient norm (coords inner, basis outer),
    mirroring the legacy ``test_chain_deriv`` oracle.
    """
    coords = np.asarray(COORDS)

    def gnorm(basis):
        def inner(coords_):
            mol = MoleLite(symbols=SYMBOLS, coords=coords_, basis=basis,
                           trace_coords=True, trace_basis=True)
            return np.linalg.norm(mol.intor("int1e_ovlp", hermi=1))
        return np.linalg.norm(jax.grad(inner)(coords))

    grad = jax.grad(gnorm)(basis)
    assert_cs_exp_close(grad, four_point_fd(gnorm, basis), 1e-6)


@pytest.mark.parametrize("intor", ["int1e_nuc", "int2e"])
def test_unsupported_intor(basis, intor):
    with pytest.raises(NotImplementedError):
        jax.jacfwd(intor_fn(intor))(basis)
