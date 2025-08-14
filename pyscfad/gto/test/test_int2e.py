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
import numpy as np
import jax
from jax import numpy as jnp

import pyscf
from pyscfad import gto

tol_val = 1e-12
tol_nuc = 1e-10
tol_cs = 1e-10
tol_exp = 1e-10

@pytest.fixture
def get_mol0():
    mol = pyscf.M(
        atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587',
        basis = '6-31g',
        verbose=0,
    )
    return mol

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = '6-31g'
    mol.verbose=0
    mol.build(trace_coords=True, trace_exp=True, trace_ctr_coeff=True)
    return mol

def int2e_grad_analyt(mol):
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    g = np.zeros((nao,nao,nao,nao,mol.natm,3))
    h1 = -mol.intor("int2e_ip1", comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        g[p0:p1,:,:,:,k] += h1[:,p0:p1].transpose(1,2,3,4,0)
        g[:,p0:p1,:,:,k] += h1[:,p0:p1].transpose(2,1,3,4,0)
        g[:,:,p0:p1,:,k] += h1[:,p0:p1].transpose(3,4,1,2,0)
        g[:,:,:,p0:p1,k] += h1[:,p0:p1].transpose(3,4,2,1,0)
    return g

def four_point_fd(mol, intor, _env_of, disp=1e-4):
    grad_fd = []
    for _, ptr_exp in enumerate(_env_of):
        #ptr_exp = _env_of[i]
        mol._env[ptr_exp] += disp
        sp = mol.intor(intor)
        mol._env[ptr_exp] += disp
        sp2 = mol.intor(intor)
        mol._env[ptr_exp] -= disp * 4.
        sm2 = mol.intor(intor)
        mol._env[ptr_exp] += disp
        sm = mol.intor(intor)
        g = (8.*(sp-sm) - sp2 + sm2) / (12.*disp)
        grad_fd.append(g)
        mol._env[ptr_exp] += disp
    return np.asarray(grad_fd).transpose(1,2,3,4,0)

def cs_grad_fd(mol, intor):
    disp = 1e-4
    _, _, _env_of = gto.mole.setup_ctr_coeff(mol)
    g = four_point_fd(mol, intor, _env_of, disp)
    return g

def exp_grad_fd(mol, intor):
    disp = 1e-4
    _, _, _env_of = gto.mole.setup_exp(mol)
    g = four_point_fd(mol, intor, _env_of, disp)
    return g

def func(mol, intor):
    return mol.intor(intor)

def func1(mol, intor):
    return jnp.linalg.norm(mol.intor(intor))

# pylint: disable=redefined-outer-name
def test_int2e(get_mol0, get_mol):
    mol0 = get_mol0
    eri0 = mol0.intor("int2e")
    mol1 = get_mol
    eri = mol1.intor("int2e")
    assert abs(eri-eri0).max() < tol_val

    tmp_nuc = int2e_grad_analyt(mol0)
    tmp_cs = cs_grad_fd(mol0, "int2e")
    tmp_exp = exp_grad_fd(mol0, "int2e")

    g0_nuc = tmp_nuc
    g0_cs = tmp_cs
    g0_exp = tmp_exp
    jac = jax.jacfwd(func)(mol1, "int2e")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    g_exp = jac.exp
    assert abs(g_nuc-g0_nuc).max() < tol_nuc
    assert abs(g_cs-g0_cs).max() < tol_cs
    assert abs(g_exp-g0_exp).max() < tol_exp

    g0_nuc = np.einsum("ijkl,ijklnx->nx", eri0, tmp_nuc) / np.linalg.norm(eri0)
    g0_cs = np.einsum("ijkl,ijklx->x", eri0, tmp_cs) / np.linalg.norm(eri0)
    g0_exp = np.einsum("ijkl,ijklx->x", eri0, tmp_exp) / np.linalg.norm(eri0)
    jac = jax.jacfwd(func1)(mol1, "int2e")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    g_exp = jac.exp
    assert abs(g_nuc-g0_nuc).max() < tol_nuc
    assert abs(g_cs-g0_cs).max() < tol_cs
    assert abs(g_exp-g0_exp).max() < tol_exp
