import pytest
import jax
import numpy as np
from pyscfad import gto

INTORS = ['int1e_kin_dr10', 'int1e_kin_dr01', 
          'int1e_ovlp_dr10', 'int1e_ovlp_dr01',
          'int1e_nuc_dr10', 'int1e_nuc_dr01',
          'int2e_dr1000', 'int2e_dr0010']

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
    grad_fd = np.asarray(grad_fd)
    grad_fd = np.moveaxis(grad_fd, 0,-1)
    return grad_fd

def cs_grad_fd(mol, intor):
    disp = 1e-3
    _, _, _env_of = gto.mole.setup_ctr_coeff(mol)
    g = four_point_fd(mol, intor, _env_of, disp)
    return g

def exp_grad_fd(mol, intor):
    disp = 1e-4
    _, _, _env_of = gto.mole.setup_exp(mol)
    g = four_point_fd(mol, intor, _env_of, disp)
    return g

def test_cs_exp():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 .74'  # in Angstrom
    mol.basis = 'sto3g'
    mol.build(trace_coords=False, trace_ctr_coeff=True, trace_exp=True)

    tol = 1e-6
    for i, intor in enumerate(INTORS):
        g_fd = cs_grad_fd(mol, intor)
        jac_fwd = jax.jacfwd(mol.__class__.intor)(mol, intor)
        e_fwd = abs(jac_fwd.ctr_coeff - g_fd).max()
        assert e_fwd < tol
        jac_rev = jax.jacrev(mol.__class__.intor)(mol, intor)
        e_rev = abs(jac_rev.ctr_coeff - g_fd).max()
        assert e_rev < tol

        g_fd = exp_grad_fd(mol, intor)
        e_fwd = abs(jac_fwd.exp - g_fd).max()
        assert e_fwd < tol
        e_rev1 = abs(jac_rev.exp - g_fd).max()
        assert e_rev < tol
