import pytest
from pyscfad import gto

@pytest.fixture(scope="module")
def get_mol():
    mol = gto.Mole()
    mol.atom = 'H 0. 0. 0.; F 0. 0. 1.1'
    mol.basis = '631g'
    mol.verbose = 0
    mol.incore_anyway = True
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol
