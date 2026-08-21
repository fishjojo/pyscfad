# Copyright 2021-2025 The PySCFAD Authors
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

"""Tests for pyscfad.scf
"""
import numpy
import pytest
import jax
from pyscfad import config_update
from pyscfad import scf
from pyscfad.gto import MoleLite
from pyscfad.scf import hf_lite
from .util import (
    hf_energy,
    df_hf_energy,
    hf_nuc_grad,
    hf_nuc_hess,
    hf_nuc_deriv3,
)

def test_rhf_nuc_grad(mol_H2O, mol_N2):
    mol = mol_H2O()
    g1 = jax.grad(hf_energy)(mol, scf.RHF).coords
    g0 = hf_nuc_grad(mol, scf.RHF)
    assert abs(g1 - g0).max() < 1e-6

    mol = mol_N2()
    g1 = jax.grad(hf_energy)(mol, scf.RHF).coords
    g0 = hf_nuc_grad(mol, scf.RHF)
    assert abs(g1 - g0).max() < 1e-6

def test_uhf_nuc_grad(mol_H2O):
    mol = mol_H2O(charge=1, spin=1)
    g1 = jax.grad(hf_energy)(mol, scf.UHF).coords
    g0 = hf_nuc_grad(mol, scf.UHF)
    assert abs(g1 - g0).max() < 1e-6

def test_rohf_nuc_grad(mol_H2O):
    mol = mol_H2O(charge=1, spin=1)
    g1 = jax.grad(hf_energy)(mol, scf.ROHF).coords
    g0 = hf_nuc_grad(mol, scf.ROHF)
    assert abs(g1 - g0).max() < 1e-6

def test_ghf_nuc_grad(mol_H2O):
    mol = mol_H2O(charge=1, spin=1)
    g1 = jax.grad(hf_energy)(mol, scf.GHF).coords
    g0 = numpy.array(
        [[0.,  0.            ,  3.27775107e-03],
         [0.,  4.31591316e-02, -1.63887553e-03],
         [0., -4.31591316e-02, -1.63887553e-03],]
    )
    assert abs(g1 - g0).max() < 1e-6

def test_rhf_nuc_hess(mol_N2):
    mol = mol_N2()
    hess1 = jax.jacrev(jax.grad(hf_energy))(mol, scf.RHF).coords.coords
    hess0 = hf_nuc_hess(mol, scf.RHF)
    assert abs(hess1 - hess0).max() < 1e-6

def test_rhf_nuc_deriv3(mol_H2):
    mol = mol_H2(basis="sto3g")
    e3 = jax.jacfwd(jax.jacfwd(jax.grad(hf_energy)))(mol, scf.RHF).coords.coords.coords
    e3_fdiff = hf_nuc_deriv3(mol, scf.RHF)
    assert abs(e3 - e3_fdiff).max() < 1e-6

def test_df_rhf_nuc_grad(mol_H2):
    mol = mol_H2()
    g0 = numpy.array([[0,0,-0.007562138], [0,0,0.007562138]])

    g1 = jax.grad(df_hf_energy)(mol, scf.RHF).coords
    assert abs(g1 - g0).max() < 1e-6

    with config_update('pyscfad_scf_implicit_diff', True):
        g1 = jax.grad(df_hf_energy)(mol, scf.RHF).coords
    assert abs(g1 - g0).max() < 1e-6

def test_df_ghf_nuc_grad(mol_H2O):
    mol = mol_H2O(charge=1, spin=1)
    g1 = jax.grad(df_hf_energy)(mol, scf.GHF).coords
    g0 = numpy.array(
        [[0.,  0.            ,  3.28370507e-03],
         [0.,  4.31595057e-02, -1.64185254e-03],
         [0., -4.31595057e-02, -1.64185254e-03],]
    )
    assert abs(g1 - g0).max() < 1e-6

def test_to_pyscf(mol_N2):
    ehf = scf.UHF(mol_N2()).density_fit().to_pyscf().kernel()
    assert abs(ehf - -108.867850114325) < 1e-8

@pytest.mark.parametrize("aosym", ["s4", "s8"])
@pytest.mark.parametrize("hermi", [1, 0])
def test_dot_eri_dm_packed(hermi, aosym):
    """The packed J/K builds reproduce the unpacked ``s1`` one, for a
    symmetric (``hermi=1``) and a general (``hermi=0``) density matrix.
    """
    mol = MoleLite(("H", "F"), [[0., 0., 0.], [0., 0., 1.1]],
                   basis="631g", verbose=0)
    eri_s1 = mol.intor("int2e", aosym="s1")
    eri = mol.intor("int2e", aosym=aosym)

    rng = numpy.random.default_rng(0)
    dm = rng.standard_normal((2, mol.nao, mol.nao))
    if hermi == 1:
        dm = dm + dm.transpose(0,2,1)

    vj0, vk0 = hf_lite.dot_eri_dm(eri_s1, dm, hermi)
    vj1, vk1 = hf_lite.dot_eri_dm(eri, dm, hermi)
    assert abs(vj1 - vj0).max() < 1e-12
    assert abs(vk1 - vk0).max() < 1e-12

    # with_j and with_k select the requested matrices only
    vj1, vk1 = hf_lite.dot_eri_dm(eri, dm, hermi, with_k=False)
    assert vk1 is None
    assert abs(vj1 - vj0).max() < 1e-12
    vj1, vk1 = hf_lite.dot_eri_dm(eri, dm, hermi, with_j=False)
    assert vj1 is None
    assert abs(vk1 - vk0).max() < 1e-12

    # the index bookkeeping must not upcast a lower working precision
    vj1, vk1 = hf_lite.dot_eri_dm(eri.astype(numpy.float32),
                                  numpy.float32(dm), hermi)
    assert vj1.dtype == numpy.float32
    assert vk1.dtype == numpy.float32

@pytest.mark.parametrize("aosym", ["s8", "s4"])
def test_rhf_lite_packed_nuc_grad(aosym):
    """SCFLite feeds a packed integral array through ``dot_eri_dm``, under
    jit and through the implicit derivative. ``get_jk`` builds an ``s8``
    array itself; any layout ``dot_eri_dm`` accepts can be preloaded into
    ``_eri`` instead.
    """
    symbols = ("O", "H", "H")
    coords = numpy.array([[0., 0., 0.23],
                          [0., 1.43, -0.92],
                          [0., -1.43, -0.92]])

    def energy(coords):
        mol = MoleLite(symbols, coords, basis="sto3g", verbose=0)
        mf = hf_lite.SCFLite(mol)
        mf._eri = mol.intor("int2e", aosym=aosym)
        mf.init_guess = "hcore"
        mf.diis = "anderson"
        return mf.kernel()

    mf0 = MoleLite(symbols, coords, basis="sto3g", verbose=0).to_pyscf().RHF()
    mf0.kernel()

    e = jax.jit(energy)(coords)
    assert abs(e - mf0.e_tot) < 1e-8

    g = jax.jit(jax.grad(energy))(coords)
    assert abs(g - mf0.nuc_grad_method().kernel()).max() < 1e-6
