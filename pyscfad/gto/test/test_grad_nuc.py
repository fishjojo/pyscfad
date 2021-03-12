import pytest
import jax
import pyscf
from pyscf.grad.rhf import grad_nuc
from pyscfad import gto

@pytest.fixture
def get_mol():
    mol = pyscf.M(
        atom = 'H 0 0 0; H 0 0 0.74',  # in Angstrom
        basis = '631g',
        verbose=0,
    )
    return mol

def test_grad_nuc(get_mol):
    mol0 = get_mol
    coords = mol0.atom_coords()
    mol = gto.Mole(mol0, coords)

    g0 = grad_nuc(mol0)

    jac = jax.jacfwd(mol.__class__.energy_nuc)(mol)
    g = jac.coords

    assert abs(g-g0).max() < 1e-10
