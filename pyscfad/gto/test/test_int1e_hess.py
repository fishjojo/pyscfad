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

TOL_NUC = 1e-8
TOL_NUC2 = 5e-8

TEST_SET = ["int1e_ovlp", "int1e_kin", "int1e_nuc",
            "int1e_rinv",]
TEST_SET_NUC = ["int1e_nuc"]

@pytest.fixture
def get_mol0():
    mol = pyscf.M(
        atom = 'H 0. 0. 0.; F 0. , 0. , 0.91',
        basis = 'ccpvdz',
        verbose=0,
    )
    return mol

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'H 0. 0. 0.; F 0. , 0. , 0.91'
    mol.basis = 'ccpvdz'
    mol.verbose=0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def hess_analyt(mol, intor):
    '''
    Hessian for int1e_ without rc derivative
    '''
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    s0 = mol.intor(intor)
    nao = s0.shape[-1]
    h = numpy.zeros((nao,nao,mol.natm,3,mol.natm,3))
    ip2 = intor.replace("int1e_","int1e_ipip")
    ipip = intor.replace("_sph","").replace("_cart","").replace("int1e_","int1e_ip")+"ip"
    if "_sph" in intor:
        ipip = ipip + "_sph"
    elif "_cart" in intor:
        ipip = ipip + "_cart"
    s2 = mol.intor(ip2).reshape(3,3,nao,nao).transpose(2,3,0,1)
    s12 = mol.intor(ipip).reshape(3,3,nao,nao).transpose(2,3,0,1)
    for ia in atmlst:
        p0, p1 = aoslices[ia,2:]
        h[p0:p1,:,ia,:,ia] += s2[p0:p1]
        h[:,p0:p1,ia,:,ia] += s2[p0:p1].transpose(1,0,2,3)
        for ja in atmlst:
            q0, q1 = aoslices[ja,2:]
            h[p0:p1,q0:q1,ia,:,ja,:] += s12[p0:p1,q0:q1]
            h[q0:q1,p0:p1,ia,:,ja,:] += s12[q0:q1,p0:p1].transpose(0,1,3,2)
    return h

def hess_analyt_nuc(mol):
    '''
    Hessian for int1e_nuc
    '''
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    nbas = mol.nbas
    h = numpy.zeros((nao,nao,mol.natm,3,mol.natm,3))

    h1aa = mol.intor('int1e_ipipnuc', comp=9).reshape(3,3,nao,nao)
    h1ab = mol.intor('int1e_ipnucip', comp=9).reshape(3,3,nao,nao)
    for iatm in atmlst:
        ish0, ish1, i0, i1 = aoslices[iatm]
        zi = mol.atom_charge(iatm)
        for jatm in atmlst:
            jsh0, jsh1, j0, j1 = aoslices[jatm]
            zj = mol.atom_charge(jatm)
            if iatm == jatm:
                with mol.with_rinv_at_nucleus(iatm):
                    rinv2aa = mol.intor('int1e_ipiprinv', comp=9)
                    rinv2ab = mol.intor('int1e_iprinvip', comp=9)
                    rinv2aa *= zi
                    rinv2ab *= zi
                    #if with_ecp and iatm in ecp_atoms:
                    #    rinv2aa -= mol.intor('ECPscalar_ipiprinv', comp=9)
                    #    rinv2ab -= mol.intor('ECPscalar_iprinvip', comp=9)
                rinv2aa = rinv2aa.reshape(3,3,nao,nao)
                rinv2ab = rinv2ab.reshape(3,3,nao,nao)
                hcore = -rinv2aa - rinv2ab
                hcore[:,:,i0:i1] += h1aa[:,:,i0:i1]
                hcore[:,:,i0:i1] += rinv2aa[:,:,i0:i1]
                hcore[:,:,i0:i1] += rinv2ab[:,:,i0:i1]
                hcore[:,:,:,i0:i1] += rinv2aa[:,:,i0:i1].transpose(0,1,3,2)
                hcore[:,:,:,i0:i1] += rinv2ab[:,:,:,i0:i1]
                hcore[:,:,i0:i1,i0:i1] += h1ab[:,:,i0:i1,i0:i1]
            else:
                hcore = numpy.zeros((3,3,nao,nao))
                hcore[:,:,i0:i1,j0:j1] += h1ab[:,:,i0:i1,j0:j1]
                with mol.with_rinv_at_nucleus(iatm):
                    shls_slice = (jsh0, jsh1, 0, nbas)
                    rinv2aa = mol.intor('int1e_ipiprinv', comp=9, shls_slice=shls_slice)
                    rinv2ab = mol.intor('int1e_iprinvip', comp=9, shls_slice=shls_slice)
                    rinv2aa *= zi
                    rinv2ab *= zi
                    #if with_ecp and iatm in ecp_atoms:
                    #    rinv2aa -= mol.intor('ECPscalar_ipiprinv', comp=9, shls_slice=shls_slice)
                    #    rinv2ab -= mol.intor('ECPscalar_iprinvip', comp=9, shls_slice=shls_slice)
                    hcore[:,:,j0:j1] += rinv2aa.reshape(3,3,j1-j0,nao)
                    hcore[:,:,j0:j1] += rinv2ab.reshape(3,3,j1-j0,nao).transpose(1,0,2,3)
                with mol.with_rinv_at_nucleus(jatm):
                    shls_slice = (ish0, ish1, 0, nbas)
                    rinv2aa = mol.intor('int1e_ipiprinv', comp=9, shls_slice=shls_slice)
                    rinv2ab = mol.intor('int1e_iprinvip', comp=9, shls_slice=shls_slice)
                    rinv2aa *= zj
                    rinv2ab *= zj
                    #if with_ecp and jatm in ecp_atoms:
                    #    rinv2aa -= mol.intor('ECPscalar_ipiprinv', comp=9, shls_slice=shls_slice)
                    #    rinv2ab -= mol.intor('ECPscalar_iprinvip', comp=9, shls_slice=shls_slice)
                    hcore[:,:,i0:i1] += rinv2aa.reshape(3,3,i1-i0,nao)
                    hcore[:,:,i0:i1] += rinv2ab.reshape(3,3,i1-i0,nao)
            hcore = hcore + hcore.conj().transpose(0,1,3,2)
            h[:,:,iatm,:,jatm] = hcore.transpose(2,3,0,1)
    return h

def _test_int1e_deriv2_nuc(intor, mol0, mol1, funanal, args, hermi=0, tol=TOL_NUC):
    hess0 = funanal(*args)
    hess_fwd = jax.jacfwd(jax.jacfwd(mol1.__class__.intor))(mol1, intor, hermi=hermi)
    hess_rev = jax.jacrev(jax.jacrev(mol1.__class__.intor))(mol1, intor, hermi=hermi)
    assert abs(hess_fwd.coords.coords - hess0).max() < tol
    assert abs(hess_rev.coords.coords - hess0).max() < tol

# pylint: disable=redefined-outer-name
def test_int1e_deriv2(get_mol0, get_mol):
    mol0 = get_mol0
    mol1 = get_mol
    for intor in set(TEST_SET) - set(TEST_SET_NUC):
        _test_int1e_deriv2_nuc(intor, mol0, mol1, hess_analyt, (mol0, intor), hermi=1)

    for intor in set(TEST_SET_NUC):
        _test_int1e_deriv2_nuc(intor, mol0, mol1, hess_analyt_nuc, (mol0,), hermi=1, tol=TOL_NUC2)
