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
parameters of a :class:`~pyscfad.ml.gto.MolePad`, i.e. w.r.t. its
:class:`~pyscfad.ml.gto.BasisArray`.

The padded counterpart of :mod:`pyscfad.gto.test.test_deriv1_cs_exp_lite`.
:class:`~pyscfad.ml.gto.MolePad` pads every
element to the same numbers of shells, primitives and contractions, so the
whole basis is one ``(n_element, nbas, nprim, 1 + nctr)`` array whose fastest
axis is ``(exponent, coefficients...)`` -- the same split as the
``(nprim, 1 + nctr)`` leaves of ``MoleLite.basis`` -- with ``mask_data``
marking the slots that carry a real parameter.

Unlike :class:`~pyscfad.gto.MoleLite`, ``_bas`` is built from the (possibly
traced) atomic numbers, so this is the only path exercising the traced-``bas``
branch of the derivative, where the static shell structure comes from
``BasisArray.metadata`` rather than from a concrete ``bas``.

Two things differ from the lite file:

- The derivative is taken w.r.t. ``BasisArray.data`` rather than the whole
  :class:`~pyscfad.ml.gto.BasisArray`, because the ``mask_*`` leaves are
  boolean and ``jax.jacfwd``/``jax.grad`` reject boolean inputs outright.
  :func:`test_cs_exp_masks_no_deriv` covers the whole pytree separately.
- The finite-difference oracle is **sampled**. The padded basis carries a few
  hundred parameters, most of them padding, and every displacement costs an
  integral evaluation.
"""
import contextlib
import dataclasses

import numpy
import pytest
import jax

from pyscfad import numpy as np
from pyscfad.ml.gto import MolePad, make_basis_array

# a water molecule plus one padded (ghost) atom
NUMBERS = numpy.array([8, 1, 1, 0], dtype=numpy.int32)
COORDS = numpy.array([[0.0, 0.0, 0.4],
                      [0.2, 1.4, -1.0],
                      [-0.2, -1.45, -0.9],
                      [0.0, 0.0, 0.0]])
# gauge origin of int1e_r / int1e_rinv, off every nucleus
ORIGIN = numpy.array([0.3, -0.2, 0.6])

BASIS = make_basis_array("sto-3g", max_number=8)

INTORS = ["int1e_ovlp", "int1e_kin", "int1e_nuc", "int1e_ovlp_dr10"]


@pytest.fixture(scope="module")
def data():
    """The differentiated parameter array of :data:`BASIS`."""
    return BASIS.data


@pytest.fixture(scope="module")
def ao_loc():
    return MolePad(NUMBERS, np.asarray(COORDS), basis=BASIS).ao_loc


@pytest.fixture(scope="module")
def slots():
    """Flat ``data`` indices the finite differences sample."""
    return sample_slots(BASIS)


def with_data(data):
    """:data:`BASIS` carrying ``data`` in place of its own."""
    return dataclasses.replace(BASIS, data=data)


def intor_fn(intor, hermi=0, cart=False, shls_slice=None, origin=None,
             numbers=NUMBERS, coords=COORDS):
    """``data -> integral`` for a fixed geometry. ``origin`` places the gauge
    origin of the operator at :data:`ORIGIN`, "common" for ``int1e_r``-type
    and "rinv" for ``int1e_rinv``-type integrals.
    """
    def fn(data):
        mol = MolePad(numbers, np.asarray(coords), basis=with_data(data),
                      cart=cart)
        if origin == "common":
            ctx = mol.with_common_origin(ORIGIN)
        elif origin == "rinv":
            ctx = mol.with_rinv_origin(ORIGIN)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            return mol.intor(intor, hermi=hermi, shls_slice=shls_slice)
    return fn


def sample_slots(basis, n_checks=6):
    """A sample of real (unmasked) flat ``data`` indices, half exponents and
    half coefficients.

    The two kinds alternate along the fastest axis, so a plain stride over the
    real slots can hit only one of them.
    """
    real = numpy.flatnonzero(numpy.asarray(basis.mask_data).ravel())
    ncol = numpy.shape(basis.data)[-1]
    out = []
    for kind in (real[real % ncol == 0], real[real % ncol != 0]):
        out.extend(kind[:: max(1, 2 * len(kind) // n_checks)][:n_checks // 2])
    return [int(i) for i in out]


def four_point_fd(fn, data, slots, disp=1e-4):
    """4-point finite differences of ``fn(data)`` w.r.t. the sampled ``slots``,
    as ``{flat index: array of shape fn(data).shape}``.
    """
    data = numpy.asarray(data, dtype=float)
    grad_fd = {}
    for flat in slots:
        d = disp * max(1.0, abs(data.flat[flat]))

        def at(x):
            data1 = data.copy()
            data1.flat[flat] += x
            return numpy.asarray(fn(np.asarray(data1)))

        sp, sm = at(d), at(-d)
        sp2, sm2 = at(2 * d), at(-2 * d)
        grad_fd[flat] = (8.0 * (sp - sm) - (sp2 - sm2)) / (12.0 * d)
    return grad_fd


def assert_cs_exp_close(jac, grad_fd, ncol, tol=1e-7):
    """Compare the sampled exponent and coefficient derivatives.

    ``jac`` has the integral shape followed by the four ``data`` axes; the
    fastest of them is ``(exponent, coefficients...)``, so a flat index is an
    exponent when it is a multiple of ``ncol``.
    """
    jac = numpy.asarray(jac)
    jac = jac.reshape(jac.shape[:-4] + (-1,))
    n_exp = n_cs = 0
    for flat, fd in grad_fd.items():
        assert abs(jac[..., flat] - fd).max() < tol
        n_exp += flat % ncol == 0
        n_cs += flat % ncol != 0
    assert n_exp and n_cs, "neither the exponent nor the coefficient part ran"


def assert_masks_no_deriv(grad):
    """The masks are structural and carry no derivative: float0 tangents."""
    for mask in (grad.mask_shl, grad.mask_ctr, grad.mask_data):
        assert numpy.asarray(mask).dtype == jax.dtypes.float0


@pytest.mark.parametrize("intor", INTORS)
def test_cs_exp(data, slots, intor):
    hermi = 0 if "_dr" in intor else 1
    fn = intor_fn(intor, hermi=hermi)

    jac = jax.jacfwd(fn)(data)
    assert numpy.isfinite(numpy.asarray(jac)).all()
    assert_cs_exp_close(jac, four_point_fd(fn, data, slots),
                        numpy.shape(data)[-1])


def test_cs_exp_fwd_rev(data):
    """Forward and reverse mode agree.

    One integral only: a padded reverse-mode jacobian is the most expensive
    thing in this file, and the transposition it checks is shared by all of
    them.
    """
    fn = intor_fn("int1e_ovlp", hermi=1)
    jac_fwd = numpy.asarray(jax.jacfwd(fn)(data))
    jac_rev = numpy.asarray(jax.jacrev(fn)(data))
    assert abs(jac_fwd - jac_rev).max() < 1e-12


@pytest.mark.parametrize("intor,origin", [("int1e_r", "common"),
                                          ("int1e_rinv", "rinv")])
def test_cs_exp_origin(data, slots, intor, origin):
    """Operators carrying a gauge origin. The origin is a plain constant in
    ``_env``, so only the basis parameters contribute to the tangent.
    """
    fn = intor_fn(intor, hermi=1, origin=origin)

    jac = jax.jacfwd(fn)(data)
    assert_cs_exp_close(jac, four_point_fd(fn, data, slots),
                        numpy.shape(data)[-1])


def test_cs_exp_cart(data, slots):
    """Cartesian AOs skip the cartesian-to-spherical transformation of the
    exponent derivative.
    """
    fn = intor_fn("int1e_ovlp", hermi=1, cart=True)

    jac = jax.jacfwd(fn)(data)
    assert_cs_exp_close(jac, four_point_fd(fn, data, slots),
                        numpy.shape(data)[-1])


# 4 atoms x 3 shells (s, s, p) with ao_loc = [0, 1, 2, 5, ...];
# hermi = 1 is only defined for a diagonal shell block
SLICES = [((1, 5, 2, 9), 0),
          ((2, 7, 2, 7), 1)]


@pytest.mark.parametrize("shls_slice,hermi", SLICES)
def test_cs_exp_shls_slice(data, ao_loc, slots, shls_slice, hermi):
    """Basis derivatives of a partial shell block. Bra and ket select
    different shell ranges, so the two cross blocks differ.
    """
    fn = intor_fn("int1e_ovlp", hermi=hermi, shls_slice=shls_slice)

    jac = jax.jacfwd(fn)(data)
    assert_cs_exp_close(jac, four_point_fd(fn, data, slots),
                        numpy.shape(data)[-1])

    # the sliced derivative is the matching block of the full one
    i0, i1, j0, j1 = shls_slice
    ao_loc = numpy.asarray(ao_loc)
    jac_full = jax.jacfwd(intor_fn("int1e_ovlp", hermi=hermi))(data)
    ref = numpy.asarray(jac_full)[ao_loc[i0]:ao_loc[i1],
                                  ao_loc[j0]:ao_loc[j1]]
    jac = numpy.asarray(jac)
    assert jac.shape == ref.shape
    assert abs(jac - ref).max() < 1e-12


def test_cs_exp_jit(data):
    """The basis derivative is jittable and jit-invariant."""
    fn = intor_fn("int1e_ovlp", hermi=1)

    jac = numpy.asarray(jax.jacfwd(fn)(data))
    jac_jit = numpy.asarray(jax.jit(jax.jacfwd(fn))(data))
    assert abs(jac - jac_jit).max() < 1e-12


def test_cs_exp_mixed_coords_basis(data, slots):
    """d/d(basis) of the coordinate-gradient norm (coords inner, basis outer)."""
    def gnorm(data):
        def inner(coords_):
            mol = MolePad(NUMBERS, coords_, basis=with_data(data))
            return np.linalg.norm(mol.intor("int1e_ovlp", hermi=1))
        return np.linalg.norm(jax.grad(inner)(np.asarray(COORDS)))

    grad = jax.grad(gnorm)(data)
    # the finite differences re-enter the nested gradient once per point, so
    # the target is jitted; the derivative itself is taken eagerly above
    assert_cs_exp_close(grad, four_point_fd(jax.jit(gnorm), data, slots),
                        numpy.shape(data)[-1], tol=1e-6)


@pytest.mark.parametrize("hermi", [1, 0])
def test_cs_exp_mixed_basis_coords(data, hermi):
    """d/d(coords) of the basis-gradient norm (basis inner, coords outer).

    The reverse order of :func:`test_cs_exp_mixed_coords_basis`: here the
    cross integrals carrying the basis tangent are themselves differentiated
    with respect to the nuclear coordinates, which is what the per-atom AO
    ranges passed to the cross evaluations are for. ``hermi = 0`` also runs
    the ket cross term.
    """
    def gnorm(coords_):
        def inner(data_):
            mol = MolePad(NUMBERS, coords_, basis=with_data(data_))
            return np.linalg.norm(mol.intor("int1e_ovlp", hermi=hermi))
        return np.linalg.norm(jax.grad(inner)(data))

    coords = np.asarray(COORDS)
    grad = numpy.asarray(jax.grad(gnorm)(coords))

    fn = jax.jit(gnorm)
    grad_fd = numpy.zeros_like(grad)
    h = 1e-4
    for idx in numpy.ndindex(grad.shape):
        def at(d):
            c = numpy.array(COORDS)
            c[idx] += d
            return float(fn(np.asarray(c)))
        grad_fd[idx] = (8 * (at(h) - at(-h)) - (at(2 * h) - at(-2 * h))) / (12 * h)

    assert abs(grad - grad_fd).max() < 1e-6
    # the ghost atom carries no derivative
    assert not grad[NUMBERS == 0].any()


def test_cs_exp_masks_no_deriv():
    """Differentiating the whole :class:`~pyscfad.ml.gto.BasisArray`: the
    boolean ``mask_*`` leaves are structural, so JAX gives them a ``float0``
    tangent and reverse mode needs ``allow_int=True``.
    """
    def loss(basis):
        mol = MolePad(NUMBERS, np.asarray(COORDS), basis=basis)
        s = mol.intor("int1e_ovlp", hermi=1)
        return np.sum(s * s)

    grad = jax.grad(loss, allow_int=True)(BASIS)
    assert_masks_no_deriv(grad)
    assert numpy.isfinite(numpy.asarray(grad.data)).all()


@pytest.mark.parametrize("intor", ["int1e_ovlp", "int1e_ovlp_dr10"])
def test_cs_exp_padded_slots(data, intor):
    """Padding slots carry no derivative at all.

    They hold placeholders (exponent 1e12, zero coefficient) that contribute
    to no integral, and their derivative would be meaningless anyway:
    ``make_bas_env`` gives a padded coefficient a ``gto_norm(l, 1e12)`` ~ 1e21
    tangent multiplying cross integrals of ~1e-19, which no backend resolves
    to that relative accuracy. ``make_bas_env`` therefore freezes them.
    """
    hermi = 1 if intor == "int1e_ovlp" else 0
    jac = numpy.asarray(jax.jacfwd(intor_fn(intor, hermi=hermi))(data))

    pad = ~numpy.asarray(BASIS.mask_data)
    assert not jac[..., pad].any()


def test_cs_exp_traced_numbers(data):
    """Basis gradients with traced atomic numbers (ML training): numbers as a
    jit argument and vmap over a batch of systems.
    """
    numbers_b = numpy.array([[8, 1, 1, 0], [7, 1, 1, 1]], dtype=numpy.int32)
    coords_b = numpy.array([
        COORDS,
        [[0.0, 0.0, 0.0], [0.0, 1.9, 0.4], [1.7, -0.8, 0.4], [-1.6, -0.9, 0.5]],
    ])

    def loss(data, numbers, coords):
        mol = MolePad(numbers, coords, basis=with_data(data))
        s = mol.intor("int1e_ovlp", hermi=1)
        return np.sum(s * s)

    grad = jax.grad(loss)

    # eager references (per system)
    g_ref = [numpy.asarray(grad(data, numbers_b[i], coords_b[i]))
             for i in range(2)]

    # numbers as a traced jit argument
    g_jit = jax.jit(grad)
    for i in range(2):
        assert abs(numpy.asarray(g_jit(data, numbers_b[i], coords_b[i]))
                   - g_ref[i]).max() < 1e-12

    # vmap over the batch of systems
    g_vmap = jax.jit(jax.vmap(grad, in_axes=(None, 0, 0)))(
        data, numbers_b, coords_b)
    for i in range(2):
        assert abs(numpy.asarray(g_vmap)[i] - g_ref[i]).max() < 1e-12


@pytest.mark.parametrize("intor", ["int2e"])
def test_unsupported_intor(data, intor):
    with pytest.raises(NotImplementedError):
        jax.jacfwd(intor_fn(intor))(data)
