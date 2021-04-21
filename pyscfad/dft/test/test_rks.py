import pytest
import numpy
from pyscfad import gto, dft

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
    mol.basis = '631g'
    mol.build(trace_coords=True)
    return mol

def test_rks_nuc_grad(get_mol):
    mol = get_mol
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    g = mf.nuc_grad_ad()
    g0 = numpy.array([[0, 0, 2.24114270e-03],
                      [0, 0, -2.24114270e-03]])
    assert abs(g-g0).max() < 1e-10
