import pytest
import numpy as np
import pyscf
import jax
from pyscfad.lib import numpy as jnp
from pyscfad import gto

@pytest.fixture
def get_mol0():
    mol = pyscf.M(
        atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587',
        basis = 'sto3g',
        verbose=0,
    )
    return mol

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = 'sto3g'
    mol.verbose=0
    mol.build()
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

def func(mol, intor):
    return mol.intor(intor)

def func1(mol, intor):
    return jnp.linalg.norm(mol.intor(intor))

def test_int2e(get_mol0, get_mol):
    mol0 = get_mol0
    eri0 = mol0.intor("int2e")
    mol1 = get_mol
    eri = mol1.intor("int2e")
    assert abs(eri-eri0).max() < 1e-10

    tmp = int2e_grad_analyt(mol0)

    g0 = tmp
    jac = jax.jacfwd(func)(mol1, "int2e")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10

    g0 = np.einsum("ijkl,ijklnx->nx", eri0, tmp) / np.linalg.norm(eri0)
    jac = jax.jacfwd(func1)(mol1, "int2e")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10
