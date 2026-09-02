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

"""Tests for pyscfad.mp
"""
import numpy
import jax
from pyscf import mp as pyscf_mp
from pyscfad import config_update
from pyscfad import scf, mp

def ump2_energy(mol):
    mf = scf.UHF(mol)
    mf.kernel()
    mymp = mp.MP2(mf)
    mymp.kernel()
    return mymp.e_tot

def pyscf_ump2(mol):
    mf = scf.UHF(mol).to_pyscf()
    mf.kernel()
    mymp = pyscf_mp.UMP2(mf)
    mymp.kernel()
    return mymp

def test_mp2_dispatch(mol_H2O):
    mol = mol_H2O(charge=1, spin=1)
    assert isinstance(mp.MP2(scf.UHF(mol)), mp.UMP2)

    mol = mol_H2O()
    assert isinstance(mp.MP2(scf.RHF(mol)), mp.RMP2)

def test_ump2_energy(mol_H2O):
    mol = mol_H2O(charge=1, spin=1)
    mf = scf.UHF(mol)
    mf.kernel()
    mymp = mp.MP2(mf)
    mymp.kernel()

    ref = pyscf_ump2(mol)
    assert abs(mymp.e_corr - ref.e_corr) < 1e-9
    assert abs(mymp.e_corr_ss - ref.e_corr_ss) < 1e-9
    assert abs(mymp.e_corr_os - ref.e_corr_os) < 1e-9
    # the MO phases may differ from PySCF's, so only compare magnitudes
    for t2, t2_ref in zip(mymp.t2, ref.t2):
        assert abs(abs(numpy.asarray(t2)) - abs(t2_ref)).max() < 1e-9

def test_ump2_frozen(mol_H2O):
    mol = mol_H2O(charge=1, spin=1)
    mf = scf.UHF(mol)
    mf.kernel()
    mymp = mp.MP2(mf, frozen=1)
    mymp.kernel()

    mf0 = scf.UHF(mol).to_pyscf()
    mf0.kernel()
    ref = pyscf_mp.UMP2(mf0, frozen=1)
    ref.kernel()
    assert abs(mymp.e_corr - ref.e_corr) < 1e-9

def test_ump2_non_canonical(mol_H2O):
    # a mean field flagged as unconverged triggers the iterative solver
    mol = mol_H2O(charge=1, spin=1)
    mf = scf.UHF(mol)
    mf.kernel()
    mf.converged = False
    mymp = mp.MP2(mf)
    mymp.kernel()
    assert mymp.converged

    ref = pyscf_ump2(mol)
    assert abs(mymp.e_corr - ref.e_corr) < 1e-7

def test_ump2_nuc_grad(mol_H2O):
    mol = mol_H2O(charge=1, spin=1)
    with config_update('pyscfad_scf_implicit_diff', True):
        g1 = jax.grad(ump2_energy)(mol).coords
    g0 = pyscf_ump2(mol).nuc_grad_method().kernel()
    assert abs(g1 - g0).max() < 1e-6
