import pytest
import numpy as np
import jax
import pyscf
from pyscfad import gto
from pyscfad.lib import numpy as jnp

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

@pytest.fixture
def get_mol_ecp0():
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

@pytest.fixture
def get_mol_ecp():
    mol = gto.Mole()
    mol.atom = '''
        Na 0. 0. 0.
        H  0.  0.  1.
    '''
    mol.basis = {'Na':'lanl2dz', 'H':'sto3g'}
    mol.ecp = {'Na':'lanl2dz'}
    mol.verbose=0
    mol.build()
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

def cs_grad_fd(mol, intor):
    disp = 0.001 / 2.
    grad_fd = []
    cs, cs_of, _env_of = gto.mole.setup_ctr_coeff(mol)
    for i in range(len(cs)):
        ptr_ctr = _env_of[i]
        mol._env[ptr_ctr] += disp
        sp = mol.intor(intor)
        mol._env[ptr_ctr] -= disp *2.
        sm = mol.intor(intor)
        g = (sp-sm) / (disp*2.)
        grad_fd.append(g)
        mol._env[ptr_ctr] += disp
    grad_fd = np.asarray(grad_fd).transpose(1,2,0)
    return grad_fd

def func(mol, intor):
    return mol.intor(intor)

def func1(mol, intor):
    return jnp.linalg.norm(mol.intor(intor))

def func2(mol, intor):
    return jnp.exp(mol.intor(intor))

def test_ovlp(get_mol0, get_mol):
    mol0 = get_mol0
    s0 = mol0.intor('int1e_ovlp')
    mol1 = get_mol
    s = mol1.intor('int1e_ovlp')
    assert abs(s-s0).max() < 1e-10

    tmp_nuc = grad_analyt(mol0, "int1e_ipovlp")
    tmp_cs = cs_grad_fd(mol0, "int1e_ovlp")

    g0_nuc = tmp_nuc
    g0_cs = tmp_cs
    jac = jax.jacfwd(func)(mol1, "int1e_ovlp")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    assert abs(g_nuc-g0_nuc).max() < 1e-10
    assert abs(g_cs-g0_cs).max() < 1e-10

    g0_nuc = np.einsum("ij,ijnx->nx", s0, tmp_nuc) / np.linalg.norm(s0)
    g0_cs = np.einsum("ij,ijx->x", s0, tmp_cs) / np.linalg.norm(s0)
    jac = jax.jacfwd(func1)(mol1, "int1e_ovlp")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    assert abs(g_nuc-g0_nuc).max() < 1e-10
    assert abs(g_cs-g0_cs).max() < 1e-10

    g0_nuc = np.einsum("ij, ijnx->ijnx", np.exp(s0), tmp_nuc)
    g0_cs = np.einsum("ij, ijx->ijx", np.exp(s0), tmp_cs)
    jac = jax.jacfwd(func2)(mol1, "int1e_ovlp")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    assert abs(g_nuc-g0_nuc).max() < 1e-10
    assert abs(g_cs-g0_cs).max() < 1e-10

def test_kin(get_mol0, get_mol):
    mol0 = get_mol0
    kin0 = mol0.intor("int1e_kin")
    mol1 = get_mol
    kin = mol1.intor("int1e_kin")
    assert abs(kin-kin0).max() < 1e-10

    tmp_nuc = grad_analyt(mol0, "int1e_ipkin")
    tmp_cs = cs_grad_fd(mol0, "int1e_kin")

    g0_nuc = tmp_nuc
    g0_cs = tmp_cs
    jac = jax.jacfwd(func)(mol1, "int1e_kin")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    assert abs(g_nuc-g0_nuc).max() < 1e-10
    assert abs(g_cs-g0_cs).max() < 1e-10

    g0_nuc = np.einsum("ij,ijnx->nx", kin0, tmp_nuc) / np.linalg.norm(kin0)
    g0_cs = np.einsum("ij,ijx->x", kin0, tmp_cs) / np.linalg.norm(kin0)
    jac = jax.jacfwd(func1)(mol1, "int1e_kin")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    assert abs(g_nuc-g0_nuc).max() < 1e-10
    assert abs(g_cs-g0_cs).max() < 1e-10

def test_nuc(get_mol0, get_mol):
    mol0 = get_mol0
    nuc0 = mol0.intor("int1e_nuc")
    mol1 = get_mol
    nuc = mol1.intor("int1e_nuc")
    assert abs(nuc-nuc0).max() < 1e-10

    tmp_nuc = nuc_grad_analyt(mol0)
    tmp_cs = cs_grad_fd(mol0, "int1e_nuc")

    g0_nuc = tmp_nuc
    g0_cs = tmp_cs
    jac = jax.jacfwd(func)(mol1, "int1e_nuc")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    assert abs(g_nuc-g0_nuc).max() < 1e-10
    assert abs(g_cs-g0_cs).max() < 1e-10

    g0_nuc = np.einsum("ij,ijnx->nx", nuc0, tmp_nuc) / np.linalg.norm(nuc0)
    g0_cs = np.einsum("ij,ijx->x", nuc0, tmp_cs) / np.linalg.norm(nuc0)
    jac = jax.jacfwd(func1)(mol1, "int1e_nuc")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    assert abs(g_nuc-g0_nuc).max() < 1e-10
    assert abs(g_cs-g0_cs).max() < 1e-10

def test_ECPscalar_nuc(get_mol_ecp0, get_mol_ecp):
    mol0 = get_mol_ecp0
    nuc0 = mol0.intor("ECPscalar")
    mol1 = get_mol_ecp
    nuc = mol1.intor("ECPscalar")
    assert abs(nuc-nuc0).max() < 1e-10

    tmp = ECPscalar_grad_analyt(mol0)

    g0 = tmp
    jac = jax.jacfwd(func)(mol1, "ECPscalar")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10

    g0 = np.einsum("ij,ijnx->nx", nuc0, tmp) / np.linalg.norm(nuc0)
    jac = jax.jacfwd(func1)(mol1, "ECPscalar")
    g = jac.coords
    assert abs(g-g0).max() < 1e-10
