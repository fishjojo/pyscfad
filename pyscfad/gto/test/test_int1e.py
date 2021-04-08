import pytest
import numpy as np
import jax
import pyscf
from pyscfad import gto
from pyscfad.lib import numpy as jnp

tol_val = 1e-12
tol_nuc = 1e-10
tol_cs = 1e-10
tol_exp = 1e-9

@pytest.fixture
def get_mol0():
    mol = pyscf.M(
        atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587',
        basis = 'ccpvdz',
        verbose=0,
    )
    return mol

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = 'ccpvdz'
    mol.verbose=0
    mol.build(trace_coords=True, trace_exp=True, trace_ctr_coeff=True)
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
    mol.build(trace_coords=True, trace_exp=True, trace_ctr_coeff=True)
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

def four_point_fd(mol, intor, _env_of, disp=1e-4):
    grad_fd = []
    for i in range(len(_env_of)):
        ptr_exp = _env_of[i]
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
    return np.asarray(grad_fd).transpose(1,2,0)

def cs_grad_fd(mol, intor):
    disp = 1e-3
    cs, cs_of, _env_of = gto.mole.setup_ctr_coeff(mol)
    g = four_point_fd(mol, intor, _env_of, disp)
    return g

def exp_grad_fd(mol, intor):
    disp = 1e-4
    es, es_of, _env_of = gto.mole.setup_exp(mol)
    g = four_point_fd(mol, intor, _env_of, disp)
    return g

def func(mol, intor):
    return mol.intor(intor)

def func1(mol, intor):
    return jnp.linalg.norm(mol.intor(intor))

def test_ovlp(get_mol0, get_mol):
    mol0 = get_mol0
    s0 = mol0.intor('int1e_ovlp')
    mol1 = get_mol
    s = mol1.intor('int1e_ovlp')
    assert abs(s-s0).max() < tol_val

    tmp_nuc = grad_analyt(mol0, "int1e_ipovlp")
    tmp_cs = cs_grad_fd(mol0, "int1e_ovlp")
    tmp_exp = exp_grad_fd(mol0, "int1e_ovlp")

    g0_nuc = tmp_nuc
    g0_cs = tmp_cs
    g0_exp = tmp_exp
    jac = jax.jacfwd(func)(mol1, "int1e_ovlp")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    g_exp = jac.exp
    assert abs(g_nuc-g0_nuc).max() < tol_nuc
    assert abs(g_cs-g0_cs).max() < tol_cs
    assert abs(g_exp-g0_exp).max() < tol_exp

    g0_nuc = np.einsum("ij,ijnx->nx", s0, tmp_nuc) / np.linalg.norm(s0)
    g0_cs = np.einsum("ij,ijx->x", s0, tmp_cs) / np.linalg.norm(s0)
    g0_exp = np.einsum("ij,ijx->x", s0, tmp_exp) / np.linalg.norm(s0)
    jac = jax.jacfwd(func1)(mol1, "int1e_ovlp")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    g_exp = jac.exp
    assert abs(g_nuc-g0_nuc).max() < tol_nuc
    assert abs(g_cs-g0_cs).max() < tol_cs
    assert abs(g_exp-g0_exp).max() < tol_exp

def test_kin(get_mol0, get_mol):
    mol0 = get_mol0
    kin0 = mol0.intor("int1e_kin")
    mol1 = get_mol
    kin = mol1.intor("int1e_kin")
    assert abs(kin-kin0).max() < tol_val

    tmp_nuc = grad_analyt(mol0, "int1e_ipkin")
    tmp_cs = cs_grad_fd(mol0, "int1e_kin")
    tmp_exp = exp_grad_fd(mol0, "int1e_kin")

    g0_nuc = tmp_nuc
    g0_cs = tmp_cs
    g0_exp = tmp_exp
    jac = jax.jacfwd(func)(mol1, "int1e_kin")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    g_exp = jac.exp
    assert abs(g_nuc-g0_nuc).max() < tol_nuc
    assert abs(g_cs-g0_cs).max() < tol_cs
    assert abs(g_exp-g0_exp).max() < tol_exp

    g0_nuc = np.einsum("ij,ijnx->nx", kin0, tmp_nuc) / np.linalg.norm(kin0)
    g0_cs = np.einsum("ij,ijx->x", kin0, tmp_cs) / np.linalg.norm(kin0)
    g0_exp = np.einsum("ij,ijx->x", kin0, tmp_exp) / np.linalg.norm(kin0)
    jac = jax.jacfwd(func1)(mol1, "int1e_kin")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    g_exp = jac.exp
    assert abs(g_nuc-g0_nuc).max() < tol_nuc
    assert abs(g_cs-g0_cs).max() < tol_cs
    assert abs(g_exp-g0_exp).max() < tol_exp

def test_nuc(get_mol0, get_mol):
    mol0 = get_mol0
    nuc0 = mol0.intor("int1e_nuc")
    mol1 = get_mol
    nuc = mol1.intor("int1e_nuc")
    assert abs(nuc-nuc0).max() < tol_val

    tmp_nuc = nuc_grad_analyt(mol0)
    tmp_cs = cs_grad_fd(mol0, "int1e_nuc")
    tmp_exp = exp_grad_fd(mol0, "int1e_nuc")

    g0_nuc = tmp_nuc
    g0_cs = tmp_cs
    g0_exp = tmp_exp
    jac = jax.jacfwd(func)(mol1, "int1e_nuc")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    g_exp = jac.exp
    assert abs(g_nuc-g0_nuc).max() < tol_nuc
    assert abs(g_cs-g0_cs).max() < tol_cs
    assert abs(g_exp-g0_exp).max() < tol_exp

    g0_nuc = np.einsum("ij,ijnx->nx", nuc0, tmp_nuc) / np.linalg.norm(nuc0)
    g0_cs = np.einsum("ij,ijx->x", nuc0, tmp_cs) / np.linalg.norm(nuc0)
    g0_exp = np.einsum("ij,ijx->x", nuc0, tmp_exp) / np.linalg.norm(nuc0)
    jac = jax.jacfwd(func1)(mol1, "int1e_nuc")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    g_exp = jac.exp
    assert abs(g_nuc-g0_nuc).max() < tol_nuc
    assert abs(g_cs-g0_cs).max() < tol_cs
    assert abs(g_exp-g0_exp).max() < tol_exp

def test_ECPscalar_nuc(get_mol_ecp0, get_mol_ecp):
    mol0 = get_mol_ecp0
    nuc0 = mol0.intor("ECPscalar")
    mol1 = get_mol_ecp
    nuc = mol1.intor("ECPscalar")
    assert abs(nuc-nuc0).max() < tol_val

    tmp_nuc = ECPscalar_grad_analyt(mol0)
    tmp_cs = cs_grad_fd(mol0, "ECPscalar")
    tmp_exp = exp_grad_fd(mol0, "ECPscalar")

    g0_nuc = tmp_nuc
    g0_cs = tmp_cs
    g0_exp = tmp_exp
    jac = jax.jacfwd(func)(mol1, "ECPscalar")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    g_exp = jac.exp
    assert abs(g_nuc-g0_nuc).max() < tol_nuc
    assert abs(g_cs-g0_cs).max() < tol_cs
    assert abs(g_exp-g0_exp).max() < tol_exp

    g0_nuc = np.einsum("ij,ijnx->nx", nuc0, tmp_nuc) / np.linalg.norm(nuc0)
    g0_cs = np.einsum("ij,ijx->x", nuc0, tmp_cs) / np.linalg.norm(nuc0)
    g0_exp = np.einsum("ij,ijx->x", nuc0, tmp_exp) / np.linalg.norm(nuc0)
    jac = jax.jacfwd(func1)(mol1, "ECPscalar")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    g_exp = jac.exp
    assert abs(g_nuc-g0_nuc).max() < tol_nuc
    assert abs(g_cs-g0_cs).max() < tol_cs
    assert abs(g_exp-g0_exp).max() < tol_exp
