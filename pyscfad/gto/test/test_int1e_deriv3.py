# Copyright 2021-2025 Xing Zhang
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

import pytest
import numpy
import jax
import pyscf
from pyscfad import gto
from .test_int1e_hess import hess_analyt_nuc

TOL_NUC = 1e-8
TOL_NUC2 = 1e-6
BOHR = 0.529177249

TEST_SET = ["int1e_ovlp", "int1e_kin", "int1e_nuc", "int1e_rinv",]
TEST_SET_NUC = ["int1e_nuc"]

@pytest.fixture
def get_mol_p():
    mol = pyscf.M(
        atom = 'H 0. 0. 0.; F 0. , 0. , 0.9101',
        basis = 'sto3g',
        verbose=0,
    )
    return mol

@pytest.fixture
def get_mol_m():
    mol = pyscf.M(
        atom = 'H 0. 0. 0.; F 0. , 0. , 0.9099',
        basis = 'sto3g',
        verbose=0,
    )
    return mol

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'H 0. 0. 0.; F 0. , 0. , 0.91'
    mol.basis = 'sto3g'
    mol.verbose=0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

def deriv3(mol, intor):
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    s0 = mol.intor(intor)
    nao = s0.shape[-1]
    natm = mol.natm
    h = numpy.zeros((nao,nao, natm,3, natm,3, natm,3))
    if "_sph" in intor:
        suffix = "_sph"
    elif "_cart" in intor:
        suffix = "_cart"
    else:
        suffix = ""
    fname = intor.replace("_sph","").replace("_cart","")
    dr30 = fname + "_dr30" + suffix
    dr21 = fname + "_dr21" + suffix
    dr12 = fname + "_dr12" + suffix
    dr03 = fname + "_dr03" + suffix
    s30 = mol.intor(dr30).reshape(3,3,3,nao,nao).transpose(3,4,0,1,2)
    s21 = mol.intor(dr21).reshape(3,3,3,nao,nao).transpose(3,4,0,1,2)
    for ia in atmlst:
        p0, p1 = aoslices[ia,2:]
        h[p0:p1,:,ia,:,ia,:,ia] += s30[p0:p1]
        h[:,p0:p1,ia,:,ia,:,ia] += s30[p0:p1].transpose(1,0,2,3,4)
        for ja in atmlst:
            q0, q1 = aoslices[ja,2:]
            h[q0:q1,p0:p1,ia,:,ja,:,ja] += s21[q0:q1,p0:p1].transpose(0,1,4,2,3)
            h[p0:p1,q0:q1,ia,:,ja,:,ja] += s21[q0:q1,p0:p1].transpose(1,0,4,2,3)
            h[p0:p1,q0:q1,ia,:,ja,:,ia] += s21[p0:p1,q0:q1].transpose(0,1,2,4,3)
            h[q0:q1,p0:p1,ia,:,ja,:,ia] += s21[p0:p1,q0:q1].transpose(1,0,2,4,3)
            h[p0:p1,q0:q1,ia,:,ia,:,ja] += s21[p0:p1,q0:q1]
            h[q0:q1,p0:p1,ia,:,ia,:,ja] += s21[p0:p1,q0:q1].transpose(1,0,2,3,4)
    return -h

def deriv3_nuc_fdiff(mol_p, mol_m, disp):
    h_p = hess_analyt_nuc(mol_p)
    h_m = hess_analyt_nuc(mol_m)
    return (h_p - h_m) / disp

def _test_int1e_deriv3_nuc(intor, mol0, mol1, funanal, args, hermi=0, tol=TOL_NUC):
    hess0 = funanal(*args)
    hess_fwd = jax.jacfwd(jax.jacfwd(jax.jacfwd(mol1.__class__.intor)))(mol1, intor, hermi=hermi)
    hess_rev = jax.jacrev(jax.jacrev(jax.jacrev(mol1.__class__.intor)))(mol1, intor, hermi=hermi)
    assert abs(hess_fwd.coords.coords.coords - hess0).max() < tol
    assert abs(hess_rev.coords.coords.coords - hess0).max() < tol

# pylint: disable=redefined-outer-name
def test_int1e_deriv2(get_mol, get_mol_p, get_mol_m):
    mol0 = mol1 = get_mol
    for intor in set(TEST_SET) - set(TEST_SET_NUC):
        _test_int1e_deriv3_nuc(intor, mol0, mol1, deriv3, (mol0, intor), hermi=1)

    mol_p = get_mol_p
    mol_m = get_mol_m
    for intor in set(TEST_SET_NUC):
        disp = 0.0002 / BOHR
        h_fdiff = deriv3_nuc_fdiff(mol_p, mol_m, disp)
        hess_fwd = jax.jacfwd(jax.jacfwd(jax.jacfwd(mol1.__class__.intor)))(mol1, intor, hermi=1)
        hess_rev = jax.jacrev(jax.jacrev(jax.jacrev(mol1.__class__.intor)))(mol1, intor, hermi=1)
        assert abs(hess_fwd.coords.coords.coords[:,:,1,2] - h_fdiff).max() < TOL_NUC2
        assert abs(hess_rev.coords.coords.coords[:,:,1,2] - h_fdiff).max() < TOL_NUC2
