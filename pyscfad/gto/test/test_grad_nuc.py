import pytest
import jax
import pyscf
from pyscf.grad.rhf import grad_nuc
from pyscfad import gto

@pytest.fixture
def get_mol0():
    mol = pyscf.M(
        atom = 'H 0 0 0; H 0 0 0.74',  # in Angstrom
        basis = '631g',
        verbose=0,
    )
    return mol

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
    mol.basis = '631g'
    mol.verbose=0
    mol.build(trace_coords=True)
    return mol

def test_grad_nuc(get_mol0, get_mol):
    mol0 = get_mol0
    g0 = grad_nuc(mol0)
    mol = get_mol
    jac = jax.grad(mol.__class__.energy_nuc)(mol)
    g = jac.coords
    assert abs(g-g0).max() < 1e-10
