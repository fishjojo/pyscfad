import pytest
import numpy as np
import pyscf
import jax
from pyscfad.lib import numpy as jnp
from pyscfad import gto

@pytest.fixture
def get_mol():
    mol = pyscf.M(
        atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587',
        basis = 'sto3g',
        verbose=0,
    )
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

def test_int2e(get_mol):
    mol = get_mol
    eri0 = mol.intor("int2e")
    x = mol.atom_coords()
    mol1 = gto.mole.Mole(mol, coords=x)
    assert abs(eri0-mol1.intor("int2e")).max() < 1e-10

    tmp = int2e_grad_analyt(mol)

    g0 = tmp
    jac = jax.jacfwd(func)(mol1, "int2e")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10

    g0 = np.einsum("ijkl,ijklnx->nx", eri0, tmp) / np.linalg.norm(eri0)
    jac = jax.jacfwd(func1)(mol1, "int2e")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10
