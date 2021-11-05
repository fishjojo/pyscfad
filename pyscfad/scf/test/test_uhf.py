import pytest
import jax
import pyscf
from pyscfad import gto, scf

@pytest.fixture
def get_mol0():
    mol = pyscf.M(
        atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587',
        basis = 'sto3g',
        spin = 1,
        charge = 1,
        verbose=0,
    )
    return mol

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = 'sto3g'
    mol.spin = 1
    mol.charge = 1
    mol.verbose=0
    mol.build(trace_coords=True)
    return mol

# pylint: disable=redefined-outer-name
def test_nuc_grad(get_mol0, get_mol):
    mol = get_mol
    mf = scf.UHF(mol)
    g = mf.energy_grad().coords

    mol0 = get_mol0
    mf0 = pyscf.scf.UHF(mol0)
    mf0.kernel()
    g0 = mf0.Gradients().grad()

    assert abs(g-g0).max() < 1e-6

def test_nuc_grad_at_converge(get_mol0, get_mol):
    mol = get_mol
    mf = scf.UHF(mol)
    mf.kernel()
    g = mf.energy_grad().coords

    mol0 = get_mol0
    mf0 = pyscf.scf.UHF(mol0)
    mf0.kernel()
    g0 = mf0.Gradients().grad()

    assert abs(g-g0).max() < 1e-6
