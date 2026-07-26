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
parameters of a :class:`~pyscfad.ml.gto.MolePad`, i.e. w.r.t. its ``BasisArray``.

:class:`~pyscfad.ml.gto.MolePad` pads every element to the same numbers of
shells, primitives and contractions, so the whole basis is one
``(n_element, nbas, nprim, 1 + nctr)`` array in which ``[..., 0]`` are the
exponents and ``[..., 1:]`` the contraction coefficients. That is the padded
counterpart of the ``MoleLite.basis`` leaves in
:mod:`pyscfad.gto.test.test_deriv1_cs_exp_lite`, with ``mask_data`` marking
the slots that carry a real parameter.

Unlike :class:`~pyscfad.gto.MoleLite`, ``_bas`` is built from the (possibly
traced) atomic numbers, so this is the only path exercising the traced-``bas``
branch of the derivative, where the static shell structure comes from
``BasisArray.metadata`` rather than from a concrete ``bas``.

The parameters live in ``BasisArray.data``. The ``mask_*`` leaves are
structural and must carry no derivative; since they are boolean, JAX gives
them a ``float0`` tangent and reverse mode needs ``allow_int=True`` (a plain
``jax.grad``/``jax.jacfwd`` rejects boolean inputs outright).
"""
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


@pytest.fixture(scope="module")
def basis():
    return make_basis_array("sto-3g", max_number=8)


def loss_fn(intor, hermi=0, cart=False, shls_slice=None):
    """``BasisArray -> scalar`` for a fixed molecule."""
    def loss(basis):
        mol = MolePad(NUMBERS, COORDS, basis=basis, cart=cart, trace_basis=True)
        ints = mol.intor(intor, hermi=hermi, shls_slice=shls_slice)
        return np.sum(ints * ints)
    return loss


def basis_tangent(basis, data_tangent):
    """Tangent of a ``BasisArray`` along ``data``. The ``mask_*`` leaves are
    structural and carry no derivative, so they take the zero-sized
    ``float0`` tangent JAX uses for non-differentiable leaves.
    """
    def zero(x):
        return numpy.zeros(numpy.shape(x), dtype=jax.dtypes.float0)
    return dataclasses.replace(basis, data=data_tangent,
                               mask_shl=zero(basis.mask_shl),
                               mask_ctr=zero(basis.mask_ctr),
                               mask_data=zero(basis.mask_data))


def assert_masks_no_deriv(grad):
    """The masks carry no derivative: float0 tangents."""
    for mask in (grad.mask_shl, grad.mask_ctr, grad.mask_data):
        assert numpy.asarray(mask).dtype == jax.dtypes.float0


def two_point_fd(loss, basis, n_checks=6, disp=1e-4):
    """2-point finite differences of ``loss(basis)`` w.r.t. a sample of the
    real (unmasked) ``data`` slots, as ``(flat index, value)``.

    Exponents and coefficients are sampled separately: the real slots
    alternate between the two, so a plain stride can hit only one kind.
    """
    data = numpy.asarray(basis.data, dtype=float)
    real = numpy.flatnonzero(numpy.asarray(basis.mask_data).ravel())
    ncol = data.shape[-1]
    grad_fd = []
    for slots in (real[real % ncol == 0], real[real % ncol != 0]):
        for flat in slots[:: max(1, 2 * len(slots) // n_checks)]:
            d = disp * max(1.0, abs(data.flat[flat]))

            def at(x):
                data1 = data.copy()
                data1.flat[flat] += x
                return float(loss(dataclasses.replace(
                    basis, data=np.asarray(data1))))

            grad_fd.append((flat, (at(d) - at(-d)) / (2.0 * d)))
    return grad_fd


def assert_cs_exp_close(grad, grad_fd, ncol, tol=1e-6):
    """Compare the sampled coefficient and exponent derivatives."""
    grad = numpy.asarray(grad).ravel()
    n_exp = n_cs = 0
    for flat, fd in grad_fd:
        assert abs(grad[flat] - fd) < tol * max(1.0, abs(fd))
        # the fastest-varying axis is (exponent, coefficients...)
        n_exp += flat % ncol == 0
        n_cs += flat % ncol != 0
    assert n_exp and n_cs, "neither part was actually checked"


@pytest.mark.parametrize(
    "intor,hermi",
    [("int1e_ovlp", 1), ("int1e_kin", 0), ("int1e_ovlp_dr10", 0)],
)
def test_cs_exp(basis, intor, hermi):
    loss = loss_fn(intor, hermi=hermi)

    grad = jax.grad(loss, allow_int=True)(basis)
    assert_masks_no_deriv(grad)
    assert numpy.isfinite(numpy.asarray(grad.data)).all()

    # forward mode along a random data direction
    v = numpy.random.default_rng(0).standard_normal(numpy.shape(basis.data))
    _, tangent = jax.jvp(loss, (basis,), (basis_tangent(basis, np.asarray(v)),))
    assert abs(float(tangent)
               - float(numpy.sum(numpy.asarray(grad.data) * v))) < 1e-10

    grad_fd = two_point_fd(loss, basis)
    assert_cs_exp_close(grad.data, grad_fd, numpy.shape(basis.data)[-1])


def test_cs_exp_cart(basis):
    loss = loss_fn("int1e_ovlp", hermi=1, cart=True)

    grad = jax.grad(loss, allow_int=True)(basis)
    assert_masks_no_deriv(grad)
    grad_fd = two_point_fd(loss, basis)
    assert_cs_exp_close(grad.data, grad_fd, numpy.shape(basis.data)[-1])


# 4 atoms x 3 shells (s, s, p) with ao_loc = [0, 1, 2, 5, ...];
# hermi = 1 is only defined for a diagonal shell block
@pytest.mark.parametrize("shls_slice,hermi", [((1, 5, 2, 9), 0),
                                              ((2, 7, 2, 7), 1)])
def test_cs_exp_shls_slice(basis, shls_slice, hermi):
    """A sliced integral gives the same basis gradient as slicing the full
    one, both taken through the padded molecule.
    """
    loss = loss_fn("int1e_ovlp", hermi=hermi, shls_slice=shls_slice)
    grad = jax.grad(loss, allow_int=True)(basis)
    assert_masks_no_deriv(grad)
    assert numpy.isfinite(numpy.asarray(grad.data)).all()

    ao_loc = numpy.asarray(
        MolePad(NUMBERS, COORDS, basis=basis, trace_basis=True).ao_loc)
    i0, i1, j0, j1 = shls_slice

    def loss_block(basis):
        mol = MolePad(NUMBERS, COORDS, basis=basis, trace_basis=True)
        ints = mol.intor("int1e_ovlp", hermi=hermi)
        ints = ints[ao_loc[i0]:ao_loc[i1], ao_loc[j0]:ao_loc[j1]]
        return np.sum(ints * ints)

    assert abs(float(loss(basis)) - float(loss_block(basis))) < 1e-12
    grad_ref = jax.grad(loss_block, allow_int=True)(basis)
    assert abs(numpy.asarray(grad.data)
               - numpy.asarray(grad_ref.data)).max() < 1e-12


def test_cs_exp_jit(basis):
    """The basis gradient is jittable and jit-invariant."""
    loss = loss_fn("int1e_ovlp", hermi=1)

    grad = jax.grad(loss, allow_int=True)(basis)
    assert_masks_no_deriv(grad)
    grad_jit = jax.jit(jax.grad(loss, allow_int=True))(basis)
    assert abs(numpy.asarray(grad.data)
               - numpy.asarray(grad_jit.data)).max() < 1e-12


def test_cs_exp_padded_slots(basis):
    """Padded exponents do not move the integrals (their coefficients are
    zero), while padded coefficients carry the true, tiny derivative.
    """
    loss = loss_fn("int1e_ovlp", hermi=1)
    grad = numpy.asarray(jax.grad(loss, allow_int=True)(basis).data)

    pad = ~numpy.asarray(basis.mask_data)
    exp_col = numpy.zeros_like(pad)
    exp_col[..., 0] = True
    assert abs(grad[pad & exp_col]).max() < 1e-12
    assert abs(grad[pad & ~exp_col]).max() < 1e-6


def test_cs_exp_traced_numbers(basis):
    """Basis gradients with traced atomic numbers (ML training): numbers as
    a jit argument and vmap over a batch of systems.
    """
    numbers_b = numpy.array([[8, 1, 1, 0], [7, 1, 1, 1]], dtype=numpy.int32)
    coords_b = numpy.array([
        COORDS,
        [[0.0, 0.0, 0.0], [0.0, 1.9, 0.4], [1.7, -0.8, 0.4], [-1.6, -0.9, 0.5]],
    ])

    def loss(basis, numbers, coords):
        mol = MolePad(numbers, coords, basis=basis, trace_basis=True)
        s = mol.intor("int1e_ovlp", hermi=1)
        return np.sum(s * s)

    grad = jax.grad(loss, allow_int=True)

    # eager references (per system)
    g_ref = [numpy.asarray(grad(basis, numbers_b[i], coords_b[i]).data)
             for i in range(2)]

    # numbers as a traced jit argument
    g_jit = jax.jit(grad)
    for i in range(2):
        assert abs(numpy.asarray(g_jit(basis, numbers_b[i], coords_b[i]).data)
                   - g_ref[i]).max() < 1e-12

    # vmap over the batch of systems
    g_vmap = jax.jit(jax.vmap(grad, in_axes=(None, 0, 0)))(
        basis, numbers_b, coords_b)
    for i in range(2):
        assert abs(numpy.asarray(g_vmap.data)[i] - g_ref[i]).max() < 1e-12
