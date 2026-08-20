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

"""One-electron integrals of a :class:`~pyscfad.ml.gto.MolePad`, the padded
counterpart of :mod:`pyscfad.gto.test.test_mole_lite`.

:class:`~pyscfad.gto.MoleLite` is the reference: its own integrals and their
origin/coordinate derivatives are checked against PySCF's analytic
``int1e_irp``/``int1e_irrp``/``int1e_iprinv`` in
:func:`pyscfad.gto.test.test_mole_lite.test_int1e_origin`, so agreement with
it pins the padded path to the same analytic values. The comparison is made on
the AOs that carry a real basis function -- the padded molecule interleaves
each atom's real and padding AOs, so ``ao_mask`` rather than a leading slice
selects them.
"""
import pytest
import numpy
import jax

from pyscfad import numpy as np
from pyscfad.gto import MoleLite
from pyscfad.ml.gto import MolePad, make_basis_array

# a water molecule plus one padded (ghost) atom
NUMBERS = numpy.array([8, 1, 1, 0], dtype=numpy.int32)
SYMBOLS = ("O", "H", "H")
COORDS = numpy.array([[0.0, 0.0, 0.4],
                      [0.2, 1.4, -1.0],
                      [-0.2, -1.45, -0.9],
                      [0.0, 0.0, 0.0]])
# gauge origin, off every nucleus
R0 = numpy.array([1.0, 0.0, 1.0])
BASIS = "sto-3g"

# int1e_r / int1e_rr take the common origin, int1e_rinv the rinv one
ORIGIN_INTORS = [("int1e_r", "common"),
                 ("int1e_rr", "common"),
                 ("int1e_rinv", "rinv")]


@pytest.fixture(scope="module")
def basis():
    return make_basis_array(BASIS, max_number=8)


@pytest.fixture(scope="module")
def ao_idx(basis):
    """AOs of the padded molecule that carry a real basis function."""
    mol = MolePad(NUMBERS, np.asarray(COORDS), basis=basis)
    return numpy.flatnonzero(numpy.asarray(mol.ao_mask))


def _with_origin(mol, r0, origin):
    if origin == "rinv":
        return mol.with_rinv_origin(r0)
    elif origin == "common":
        return mol.with_common_origin(r0)
    raise NotImplementedError(origin)


def pad_fn(basis, ao_idx, intor, hermi, origin):
    """``(coords, R0) -> padded integral`` restricted to the real AOs."""
    def fn(coords, r0):
        mol = MolePad(NUMBERS, coords, basis=basis)
        with _with_origin(mol, r0, origin):
            ints = mol.intor(intor, hermi=hermi)
        return ints[..., ao_idx[:, None], ao_idx[None, :]]
    return fn


def lite_fn(intor, hermi, origin):
    """``(coords, R0) -> integral`` of the unpadded reference molecule.

    ``coords`` keeps the padded shape so that the coordinate jacobians of the
    two paths are directly comparable; the ghost row simply goes unused.
    """
    def fn(coords, r0):
        mol = MoleLite(symbols=SYMBOLS, coords=coords[:len(SYMBOLS)],
                       basis=BASIS)
        with _with_origin(mol, r0, origin):
            return mol.intor(intor, hermi=hermi)
    return fn


@pytest.mark.parametrize("intor,origin", ORIGIN_INTORS)
@pytest.mark.parametrize("hermi", [0, 1])
def test_int1e_origin(basis, ao_idx, intor, origin, hermi):
    """Integrals carrying a gauge origin, and their derivatives with respect
    to that origin and to the nuclear coordinates.

    The origin is a differentiated primal of its own, so ``d/dR0`` exercises a
    path that the basis and coordinate derivatives do not.
    """
    fn_pad = pad_fn(basis, ao_idx, intor, hermi, origin)
    fn_lite = lite_fn(intor, hermi, origin)

    coords = np.asarray(COORDS)
    r0 = np.asarray(R0)

    val_pad = numpy.asarray(fn_pad(coords, r0))
    val_lite = numpy.asarray(fn_lite(coords, r0))
    assert val_pad.shape == val_lite.shape
    assert abs(val_pad - val_lite).max() < 1e-10

    dR0_pad = numpy.asarray(jax.jacfwd(fn_pad, 1)(coords, r0))
    dR0_lite = numpy.asarray(jax.jacfwd(fn_lite, 1)(coords, r0))
    assert abs(dR0_pad - dR0_lite).max() < 1e-10
    assert abs(dR0_lite).max() > 1e-3, "the origin derivative is trivially zero"

    dR_pad = numpy.asarray(jax.jacfwd(fn_pad, 0)(coords, r0))
    dR_lite = numpy.asarray(jax.jacfwd(fn_lite, 0)(coords, r0))
    assert abs(dR_pad - dR_lite).max() < 1e-10
    assert abs(dR_lite).max() > 1e-3, "the coordinate derivative is trivially zero"
    # the ghost atom moves no basis function
    assert not dR_pad[..., NUMBERS == 0, :].any()
