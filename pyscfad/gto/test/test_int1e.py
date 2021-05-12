import pytest
import numpy as np
import jax
import pyscf
from pyscfad import gto
from pyscfad.lib import numpy as jnp

TOL_VAL = 1e-12
TOL_NUC = 1e-10
TOL_CS = 1e-10
TOL_EXP = 1e-9

TEST_SET = ["int1e_ovlp", "int1e_kin", "int1e_nuc",]
TEST_SET_ECP = ["ECPscalar"]
TEST_SET_NUC = ["int1e_nuc"]

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

def _test_int1e_value(intor, mol0, mol1, tol=TOL_VAL):
    v0 = mol0.intor(intor)
    v1 = mol1.intor(intor)
    assert abs(v1-v0).max() < tol

def _test_int1e_deriv_fd(intor, mol0, mol1, funfd, jacattr, tol):
    v0 = mol0.intor(intor)
    jac_fd = funfd(mol0, intor)
    jac_fwd = jax.jacfwd(func)(mol1, intor)
    jac_rev = jax.jacrev(func)(mol1, intor)
    assert abs(getattr(jac_fwd, jacattr) - jac_fd).max() < tol
    assert abs(getattr(jac_rev, jacattr) - jac_fd).max() < tol

    g0 = np.einsum("ij,ijx->x", v0, jac_fd) / np.linalg.norm(v0)
    jac_fwd = jax.jacfwd(func1)(mol1, intor)
    jac_rev = jax.jacrev(func1)(mol1, intor)
    assert abs(getattr(jac_fwd, jacattr) - g0).max() < tol
    assert abs(getattr(jac_rev, jacattr) - g0).max() < tol

def _test_int1e_deriv_cs(intor, mol0, mol1, tol=TOL_CS):
    _test_int1e_deriv_fd(intor, mol0, mol1,
                         cs_grad_fd, "ctr_coeff", tol)

def _test_int1e_deriv_exp(intor, mol0, mol1, tol=TOL_EXP):
    _test_int1e_deriv_fd(intor, mol0, mol1,
                         exp_grad_fd, "exp", tol)

def _test_int1e_deriv_nuc(intor, mol0, mol1, funanal, args, tol=TOL_NUC):
    v0 = mol0.intor(intor)
    jac0 = funanal(*args)
    jac_fwd = jax.jacfwd(func)(mol1, intor)
    jac_rev = jax.jacrev(func)(mol1, intor)
    assert abs(jac_fwd.coords - jac0).max() < tol
    assert abs(jac_rev.coords - jac0).max() < tol

    g0 = np.einsum("ij,ijnx->nx", v0, jac0) / np.linalg.norm(v0)
    jac_fwd = jax.jacfwd(func1)(mol1, intor)
    jac_rev = jax.jacrev(func1)(mol1, intor)
    assert abs(jac_fwd.coords - g0).max() < tol
    assert abs(jac_rev.coords - g0).max() < tol

def test_int1e(get_mol0, get_mol, get_mol_ecp0, get_mol_ecp):
    mol0 = get_mol0
    mol1 = get_mol
    for intor in TEST_SET:
        _test_int1e_value(intor, mol0, mol1)
        _test_int1e_deriv_cs(intor, mol0, mol1)
        _test_int1e_deriv_exp(intor, mol0, mol1)

    for intor in set(TEST_SET) - set(TEST_SET_NUC):
        _test_int1e_deriv_nuc(intor, mol0, mol1, grad_analyt, 
                              (mol0, intor.replace("int1e_", "int1e_ip")))
    
    for intor in TEST_SET_NUC:
        _test_int1e_deriv_nuc(intor, mol0, mol1, nuc_grad_analyt, (mol0,))

    molecp0 = get_mol_ecp0
    molecp1 = get_mol_ecp
    for intor in TEST_SET_ECP:
        _test_int1e_value(intor, molecp0, molecp1)
        _test_int1e_deriv_cs(intor, molecp0, molecp1)
        _test_int1e_deriv_exp(intor, molecp0, molecp1)
        _test_int1e_deriv_nuc(intor, molecp0, molecp1, ECPscalar_grad_analyt, (molecp0,))
