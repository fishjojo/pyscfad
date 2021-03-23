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

@pytest.fixture
def get_mol_ecp():
    mol = pyscf.M(
        atom = '''
            Na 0. 0. 0.
            H  0.  0.  1.
        ''',
        basis = {'Na':'lanl2dz', 'H':'sto3g'},
        ecp = {'Na':'lanl2dz'},
        verbose=0,
    )
    return mol

def grad_analyt(mol, intor):
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    g = np.zeros((nao,nao,mol.natm,3))
    s1 = -mol.intor(intor, comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        g[p0:p1,:,k] += s1[:,p0:p1].transpose(1,2,0)
        g[:,p0:p1,k] += s1[:,p0:p1].transpose(2,1,0)
    return g

def nuc_grad_analyt(mol):
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    g = np.zeros((nao,nao,mol.natm,3))
    h1 = -mol.intor("int1e_ipnuc", comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        with mol.with_rinv_at_nucleus(ia):
            vrinv = mol.intor('int1e_iprinv', comp=3)
            vrinv *= -mol.atom_charge(ia)
        vrinv[:,p0:p1] += h1[:,p0:p1]
        g[:,:,k] = vrinv.transpose(1,2,0) + vrinv.transpose(2,1,0)
    return g

def ECPscalar_grad_analyt(mol):
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    g = np.zeros((nao,nao,mol.natm,3))
    h1 = -mol.intor("ECPscalar_ipnuc", comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        with mol.with_rinv_at_nucleus(ia):
            vrinv = mol.intor('ECPscalar_iprinv', comp=3)
        vrinv[:,p0:p1] += h1[:,p0:p1]
        g[:,:,k] = vrinv.transpose(1,2,0) + vrinv.transpose(2,1,0)
    return g

def func(mol, intor):
    return mol.intor(intor)

def func1(mol, intor):
    return jnp.linalg.norm(mol.intor(intor))

def func2(mol, intor):
    return jnp.exp(mol.intor(intor))

def test_ovlp(get_mol):
    mol = get_mol
    s0 = mol.intor('int1e_ovlp')
    x = mol.atom_coords()
    mol1 = gto.mole.Mole(mol, coords=x)
    assert abs(s0-mol1.intor("int1e_ovlp")).max() < 1e-10

    tmp = grad_analyt(mol, "int1e_ipovlp")

    g0 = tmp
    jac = jax.jacfwd(func)(mol1, "int1e_ovlp")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10

    g0 = np.einsum("ij,ijnx->nx", s0, tmp) / np.linalg.norm(s0)
    jac = jax.jacfwd(func1)(mol1, "int1e_ovlp")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10

    g0 = np.einsum("ij, ijnx->ijnx", np.exp(s0), tmp)
    jac = jax.jacfwd(func2)(mol1, "int1e_ovlp")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10

def test_kin(get_mol):
    mol = get_mol
    kin0 = mol.intor("int1e_kin")
    x = mol.atom_coords()
    mol1 = gto.mole.Mole(mol, coords=x)
    assert abs(kin0-mol1.intor("int1e_kin")).max() < 1e-10

    tmp = grad_analyt(mol, "int1e_ipkin")

    g0 = tmp
    jac = jax.jacfwd(func)(mol1, "int1e_kin")
    g = jac.coords
    assert abs(g - g0).max() < 1e-10

    g0 = np.einsum("ij,ijnx->nx", kin0, tmp) / np.linalg.norm(kin0)
    jac = jax.jacfwd(func1)(mol1, "int1e_kin")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10

def test_nuc(get_mol):
    mol = get_mol
    nuc0 = mol.intor("int1e_nuc")
    x = mol.atom_coords()
    mol1 = gto.mole.Mole(mol, coords=x)
    assert abs(nuc0-mol1.intor("int1e_nuc")).max() < 1e-10

    tmp = nuc_grad_analyt(mol)

    g0 = tmp
    jac = jax.jacfwd(func)(mol1, "int1e_nuc")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10

    g0 = np.einsum("ij,ijnx->nx", nuc0, tmp) / np.linalg.norm(nuc0)
    jac = jax.jacfwd(func1)(mol1, "int1e_nuc")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10

def test_ECPscalar(get_mol_ecp):
    mol = get_mol_ecp
    nuc0 = mol.intor("ECPscalar")
    x = mol.atom_coords()
    mol1 = gto.mole.Mole(mol, coords=x)
    assert abs(nuc0-mol1.intor("ECPscalar")).max() < 1e-10

    tmp = ECPscalar_grad_analyt(mol)

    g0 = tmp
    jac = jax.jacfwd(func)(mol1, "ECPscalar")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10

    g0 = np.einsum("ij,ijnx->nx", nuc0, tmp) / np.linalg.norm(nuc0)
    jac = jax.jacfwd(func1)(mol1, "ECPscalar")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10
