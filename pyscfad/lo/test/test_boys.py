import pytest
import jax
from jax import numpy as np
from pyscf.data.nist import BOHR
from pyscfad import gto, scf
from pyscfad.lo import boys
from pyscfad.lo.boys import dipole_integral
from pyscfad import config

def cost_function(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    orbocc = mf.mo_coeff[:,mf.mo_occ>1e-6]
    mo_coeff = boys.boys(mol, orbocc, init_guess='atomic')

    dip = dipole_integral(mol, mo_coeff)
    r2 = mol.intor_symmetric('int1e_r2')
    r2 = np.einsum('pi,pi->', mo_coeff, np.dot(r2, mo_coeff))
    val = r2 - np.einsum('xii,xii->', dip, dip)
    return val

@pytest.fixture
def get_mol():
    config.update('pyscfad_scf_implicit_diff', True)
    #config.update('pyscfad_moleintor_opt', True)
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = '631G'
    mol.verbose = 0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    yield mol
    config.reset()

def test_boys(get_mol):
    mol = get_mol
    g0 = jax.grad(cost_function)(mol).coords

    mol.set_geom_('O 0. 0.  0.001; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587')
    f1 = cost_function(mol)
    mol.set_geom_('O 0. 0. -0.001; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587')
    f2 = cost_function(mol)
    g1 = (f1 - f2) / (0.002 / BOHR)
    assert abs(g0[0,2]-g1) < 1e-4

    mol.set_geom_('O 0. 0. 0.; H 0. , -0.756 , 0.587; H 0. , 0.757 , 0.587')
    f1 = cost_function(mol)
    mol.set_geom_('O 0. 0. 0.; H 0. , -0.758 , 0.587; H 0. , 0.757 , 0.587')
    f2 = cost_function(mol)
    g1 = (f1 - f2) / (0.002 / BOHR)
    assert abs(g0[1,1]-g1) < 1e-4

    mol.set_geom_('O 0. 0. 0.; H 0. , -0.757 , 0.588; H 0. , 0.757 , 0.587')
    f1 = cost_function(mol)
    mol.set_geom_('O 0. 0. 0.; H 0. , -0.757 , 0.586; H 0. , 0.757 , 0.587')
    f2 = cost_function(mol)
    g1 = (f1 - f2) / (0.002 / BOHR)
    assert abs(g0[1,2]-g1) < 1e-4
