# Copyright 2021-2026 The PySCFAD Authors
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

import numpy
import pytest
import jax
from pyscfad import scf, cc
from pyscfad.gto import MoleLite
from pyscfad.scf import hf_lite
from pyscfad.cc import rccsd
from pyscfad.ml.gto import MolePad, make_basis_array
from pyscfad.ml.scf import SCFPad

def test_nuc_grad(get_mol):
    mol = get_mol
    def energy(mol):
        mf = scf.RHF(mol)
        mf.kernel()
        mycc = cc.RCCSD(mf)
        mycc.kernel()
        return mycc.e_tot
    g1 = jax.grad(energy)(mol).coords
    g0 = numpy.array([[0., 0., -0.0873564848],
                      [0., 0.,  0.0873564848]])
    assert(abs(g1-g0).max() < 1e-6)

def test_df_nuc_grad(get_mol):
    mol = get_mol
    def energy(mol):
        mf = scf.RHF(mol).density_fit()
        mf.kernel()
        mycc = cc.dfccsd.RCCSD(mf)
        mycc.kernel()
        return mycc.e_tot
    g1 = jax.grad(energy)(mol).coords
    # finite difference
    g0 = numpy.array([[0., 0., -0.0873569023],
                      [0., 0.,  0.0873569023]])
    assert(abs(g1-g0).max() < 5e-6)

# ---------------------------------------------------------------------------
# Lightweight (fully jittable) implementation
# ---------------------------------------------------------------------------

# in Bohr, so the gradient is directly comparable to the PySCF one
LITE_SYMBOLS = ('H', 'F')
LITE_COORDS = numpy.array([[0., 0., 0.], [0., 0., 2.1]])
LITE_BASIS = '631g'
PAD_BASIS = LITE_BASIS
# the same molecule padded with one ghost atom
PAD_NUMBERS = numpy.array([1, 9, 0], dtype=numpy.int32)
PAD_COORDS = numpy.vstack([LITE_COORDS, [0., 0., 8.]])
PAD_NOCC = 5
# H2 with two ghost atoms: a cheap system with fake MOs to spare, so nocc can
# be pinned well above the true occupation of one
SMALL_NUMBERS = numpy.array([1, 1, 0, 0], dtype=numpy.int32)
SMALL_COORDS = numpy.array([[0., 0., 0.], [0., 0., 1.4],
                            [0., 0., 8.], [0., 0., 12.]])
# tight enough that the mean field and the amplitudes are not the error floor
SCF_TOL = 1e-12
CC_TOL = 1e-11
CC_TOL_NORMT = 1e-9


def _pyscf_ref(basis):
    """PySCF RCCSD total energy and nuclear gradient of the same molecule."""
    mol = MoleLite(LITE_SYMBOLS, LITE_COORDS, basis=basis, verbose=0).to_pyscf()
    mycc = mol.RHF().run(conv_tol=SCF_TOL).CCSD()
    mycc.conv_tol = CC_TOL
    mycc.conv_tol_normt = CC_TOL_NORMT
    mycc.run()
    return mycc.e_tot, mycc.nuc_grad_method().kernel()


def _lite_energy(coords, basis=LITE_BASIS, **kwargs):
    mol = MoleLite(LITE_SYMBOLS, coords, basis=basis, verbose=0)
    mf = hf_lite.SCFLite(mol)
    mf.init_guess = 'hcore'
    mf.diis = 'anderson'
    mf.conv_tol = SCF_TOL
    mf.kernel()
    mycc = rccsd.RCCSDLite(mf, conv_tol=CC_TOL,
                           conv_tol_normt=CC_TOL_NORMT, **kwargs)
    mycc.kernel()
    return mycc.e_tot


def _pad_energy(coords, basis, numbers=PAD_NUMBERS, nocc=PAD_NOCC, **kwargs):
    mol = MolePad(numbers, coords, basis=basis, verbose=0)
    mf = SCFPad(mol)
    mf.init_guess = 'hcore'
    mf.diis = 'anderson'
    mf.conv_tol = SCF_TOL
    mf.kernel()
    mycc = rccsd.RCCSDLite(mf, nocc=nocc, conv_tol=CC_TOL,
                           conv_tol_normt=CC_TOL_NORMT, **kwargs)
    mycc.kernel()
    return mycc.e_tot


@pytest.fixture(scope='module')
def pyscf_ref():
    return _pyscf_ref(LITE_BASIS)


@pytest.fixture(scope='module')
def pyscf_ref_pad():
    return _pyscf_ref(PAD_BASIS)


@pytest.fixture(scope='module')
def pad_basis():
    # built outside any trace: make_basis_array builds jax arrays, which
    # inside a jit would be tracers of that trace
    return make_basis_array(PAD_BASIS, max_number=9)


@pytest.fixture(scope='module')
def small_pad_basis():
    return make_basis_array(PAD_BASIS, max_number=1)


@pytest.fixture(scope='module')
def lite_result():
    return jax.jit(jax.value_and_grad(_lite_energy))(LITE_COORDS)


@pytest.fixture(scope='module')
def pad_result(pad_basis):
    fn = jax.value_and_grad(lambda c: _pad_energy(c, pad_basis))
    return jax.jit(fn)(PAD_COORDS)


def test_rccsd_lite_nuc_grad(pyscf_ref, lite_result):
    """The lite CCSD energy and its nuclear gradient, under jit. The gradient
    is the implicit derivative of the amplitude equations, not an unrolled
    iteration."""
    e_ref, g_ref = pyscf_ref
    e, g = lite_result
    assert abs(e - e_ref) < 1e-9
    assert abs(g - g_ref).max() < 1e-6


@pytest.mark.parametrize('mixer', [None, 'anderson'])
def test_rccsd_lite_mixers(pyscf_ref, mixer):
    """The fixed point does not depend on the amplitude mixer. DIIS, the
    default, is exercised by every other test here; Anderson mixing and the
    bare iterations must land on the same solution."""
    e_ref = pyscf_ref[0]
    e = jax.jit(lambda c: _lite_energy(c, diis=mixer, max_cycle=300))(LITE_COORDS)
    assert abs(e - e_ref) < 1e-8


@pytest.mark.parametrize('aosym', ['s4', 's8'])
def test_restore_eri_s1(aosym):
    """The packed AO integral layouts the lite mean field stores -- ``get_jk``
    builds an ``s8`` array -- unpack to the full tensor the integral
    transformation needs."""
    mol = MoleLite(LITE_SYMBOLS, LITE_COORDS, basis=LITE_BASIS, verbose=0)
    eri0 = mol.intor('int2e', aosym='s1')
    eri1 = rccsd.restore_eri_s1(mol.intor('int2e', aosym=aosym), mol.nao)
    assert abs(eri1 - eri0).max() < 1e-12


def test_rccsd_pad_nuc_grad(pyscf_ref_pad, pad_result):
    """A padded molecule reproduces the unpadded energy and gradient: the
    ghost atom and the fake orbitals contribute nothing."""
    e_ref, g_ref = pyscf_ref_pad
    e, g = pad_result
    assert abs(e - e_ref) < 1e-9
    assert abs(g[:len(g_ref)] - g_ref).max() < 1e-6
    assert abs(g[len(g_ref):]).max() == 0.


def test_rccsd_pad_occupied_padding(small_pad_basis):
    """``nocc`` may be pinned above the true occupation -- as a batch of padded
    molecules requires -- and the extra occupied slots are then filled with
    fake orbitals, which leaves the energy unchanged."""
    def energy(nocc):
        return _pad_energy(SMALL_COORDS, small_pad_basis,
                           numbers=SMALL_NUMBERS, nocc=nocc)

    e0 = energy(1)
    for nocc in (2, 3):
        assert abs(energy(nocc) - e0) < 1e-9


def test_rccsd_pad_nocc_requires_static(pad_basis):
    """A traced electron count cannot give a static ``nocc``."""
    def get_nocc(coords):
        mol = MolePad(PAD_NUMBERS, coords, basis=pad_basis, verbose=0)
        mf = SCFPad(mol)
        mf.mo_coeff = numpy.zeros((mol.nao, mol.nao))
        mf.mo_occ = numpy.zeros(mol.nao)
        return rccsd.RCCSDLite(mf).nocc

    assert get_nocc(PAD_COORDS) == PAD_NOCC
    with pytest.raises(TypeError, match='nocc'):
        jax.jit(get_nocc)(PAD_COORDS)
