import pytest
from functools import partial
import numpy as np
import jax
from jax import numpy as jnp
from pyscfad import gto
from pyscfad.gto._mole_helper import setup_exp, setup_ctr_coeff
from .test_int1e import grad_analyt, nuc_grad_analyt

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

def two_point_fd(mol, fn, _env_of, disp=1e-4):
    grad_fd = []
    for _, ptr_exp in enumerate(_env_of):
        mol._env[ptr_exp] += disp
        sp = fn(mol)

        mol._env[ptr_exp] -= 2*disp
        sm = fn(mol)

        g = (sp-sm) / (2*disp)
        grad_fd.append(g)
        mol._env[ptr_exp] += disp

    grad_fd = np.asarray(grad_fd)
    grad_fd = np.moveaxis(grad_fd, 0, -1)
    return grad_fd

def cs_grad_fd1(mol, fn):
    disp = 1e-3
    _, _, _env_of = setup_ctr_coeff(mol)
    g = two_point_fd(mol, fn, _env_of, disp)
    return g

def exp_grad_fd1(mol, fn):
    disp = 1e-4
    _, _, _env_of = setup_exp(mol)
    g = two_point_fd(mol, fn, _env_of, disp)
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

def test_chain_deriv():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 .74'  # in Angstrom
    mol.basis = 'sto3g'
    mol.build()

    def grad_loss_ad(mol, intor):
        def fn(mol):
            ints = mol.intor(intor)
            return jnp.linalg.norm(ints)
        g = jax.grad(fn)(mol).coords
        g_norm = jnp.linalg.norm(g)
        return g_norm

    def grad_loss_analyt(mol, intor):
        ints = mol.intor(intor)
        jac = grad_analyt(mol, intor.replace('int1e_', 'int1e_ip'))
        g = np.einsum("ij,ijnx->nx", ints, jac) / np.linalg.norm(ints)
        g_norm = np.linalg.norm(g)
        return g_norm

    def nuc_grad_loss_analyt(mol):
        ints = mol.intor('int1e_nuc')
        jac = nuc_grad_analyt(mol)
        g = np.einsum("ij,ijnx->nx", ints, jac) / np.linalg.norm(ints)
        g_norm = np.linalg.norm(g)
        return g_norm

    tol = 1e-6

    for intor in ['int1e_kin', 'int1e_ovlp']:
        grad_cs = cs_grad_fd1(mol, partial(grad_loss_analyt, intor=intor))
        grad_exp = exp_grad_fd1(mol, partial(grad_loss_analyt, intor=intor))
        jac = jax.grad(grad_loss_ad)(mol, intor)
        assert abs(jac.ctr_coeff - grad_cs).max() < tol
        assert abs(jac.exp - grad_exp).max() < tol

    grad_cs = cs_grad_fd1(mol, nuc_grad_loss_analyt)
    grad_exp = exp_grad_fd1(mol, nuc_grad_loss_analyt)
    jac = jax.grad(grad_loss_ad)(mol, 'int1e_nuc')
    assert abs(jac.ctr_coeff - grad_cs).max() < tol
    assert abs(jac.exp - grad_exp).max() < tol
