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


# ---------------------------------------------------------------------------
# padded primitive slots
# ---------------------------------------------------------------------------

# 6-31G is the case that matters below: unlike STO-3G its shells hold
# different numbers of primitives, so a *real* shell carries padded primitive
# slots.
PRIM_PADDED_BASIS = "631g"


def test_padded_primitive_exponents():
    """Every padded primitive slot repeats its shell's most diffuse real
    exponent, and a shell with no real primitive at all keeps 1.0.

    libcint bounds the prefactor of a whole shell pair from the *last* stored
    exponent of each shell, taking the exponents to be in descending order so
    that the last one is the most diffuse. Repeating the shell minimum keeps
    the stored sequence non-increasing, so the bound stays an upper one.
    """
    basis = make_basis_array(PRIM_PADDED_BASIS, max_number=8)
    exps = numpy.asarray(basis.data)[..., 0]
    real = numpy.asarray(basis.mask_data)[..., 0]
    assert (~real).any(), "no padded primitive slot to check"

    for z in range(exps.shape[0]):
        for ish in range(exps.shape[1]):
            # descending order, which is what the libcint bound assumes
            assert (numpy.diff(exps[z, ish]) <= 1e-12).all()

            padded = exps[z, ish, ~real[z, ish]]
            if not padded.size:      # a shell that fills every slot
                continue
            if real[z, ish].any():
                e_min = exps[z, ish, real[z, ish]].min()
                assert abs(padded - e_min).max() < 1e-12
            else:
                assert (padded == 1.).all()


@pytest.mark.parametrize("basis_name", [BASIS, PRIM_PADDED_BASIS])
def test_int2e(basis_name):
    """The padded two-electron integrals reproduce the unpadded ones.

    A padded primitive whose exponent sat above the real ones used to make the
    shell-pair bound too small, screening the *real* primitive pairs away along
    with the padded ones -- silently, and only in the two-electron integrals.
    """
    basis = make_basis_array(basis_name, max_number=8)
    mol_pad = MolePad(NUMBERS, np.asarray(COORDS), basis=basis)
    mol_lite = MoleLite(SYMBOLS, np.asarray(COORDS[:len(SYMBOLS)]), basis=basis_name)
    idx = numpy.flatnonzero(numpy.asarray(mol_pad.ao_mask))

    eri_lite = numpy.asarray(mol_lite.intor("int2e", aosym="s1"))
    eri_pad = numpy.asarray(mol_pad.intor("int2e", aosym="s1"))
    eri_pad = eri_pad[numpy.ix_(idx, idx, idx, idx)]
    assert eri_pad.shape == eri_lite.shape
    assert abs(eri_pad - eri_lite).max() < 1e-10


# ---------------------------------------------------------------------------
# padded contraction columns
# ---------------------------------------------------------------------------

# cc-pVDZ is generally contracted, and unevenly so: fluorine's first s shell
# carries two contractions while its remaining four shells carry one each.
# The number of contractions is therefore a property of a shell, not of an
# element.
GENERAL_CTR_BASIS = "ccpvdz"
# hydrogen alone already shows it -- its s shell is generally contracted, its
# p shell is not -- and keeps the padded 2e tensor small
H2_NUMBERS = numpy.array([1, 1, 0], dtype=numpy.int32)
H2_SYMBOLS = ("H", "H")
H2_COORDS = numpy.array([[0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.4],
                         [0.0, 0.0, 8.0]])


@pytest.mark.parametrize("basis_name", [BASIS, PRIM_PADDED_BASIS, GENERAL_CTR_BASIS])
def test_ao_mask_counts_real_aos(basis_name):
    """``ao_mask`` flags exactly as many AOs as the unpadded molecule has, and
    every one of them carries a real basis function.

    A shell with a single contraction must not inherit a second, all-zero one
    from a generally contracted sibling shell: that would be an identically
    zero basis function flagged as real, which makes the overlap of the flagged
    block singular and the mean field NaN.
    """
    basis = make_basis_array(basis_name, max_number=8)
    mol_pad = MolePad(NUMBERS, np.asarray(COORDS), basis=basis)
    mol_lite = MoleLite(SYMBOLS, np.asarray(COORDS[:len(SYMBOLS)]), basis=basis_name)

    idx = numpy.flatnonzero(numpy.asarray(mol_pad.ao_mask))
    assert len(idx) == mol_lite.nao

    ovlp = numpy.asarray(mol_pad.intor("int1e_ovlp", hermi=1))[numpy.ix_(idx, idx)]
    assert numpy.linalg.eigvalsh(ovlp).min() > 1e-6


def test_int2e_general_contraction():
    """The padded two-electron integrals of a generally contracted basis match
    the unpadded ones, which also pins the AO ordering of the flagged block."""
    basis = make_basis_array(GENERAL_CTR_BASIS, max_number=1)
    mol_pad = MolePad(H2_NUMBERS, np.asarray(H2_COORDS), basis=basis)
    mol_lite = MoleLite(H2_SYMBOLS, np.asarray(H2_COORDS[:len(H2_SYMBOLS)]),
                        basis=GENERAL_CTR_BASIS)
    idx = numpy.flatnonzero(numpy.asarray(mol_pad.ao_mask))
    assert len(idx) == mol_lite.nao

    eri_lite = numpy.asarray(mol_lite.intor("int2e", aosym="s1"))
    eri_pad = numpy.asarray(mol_pad.intor("int2e", aosym="s1"))
    eri_pad = eri_pad[numpy.ix_(idx, idx, idx, idx)]
    assert abs(eri_pad - eri_lite).max() < 1e-10
