import pytest
from pyscfad import gto, scf

@pytest.fixture
def get_h2o():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = 'sto3g'
    mol.verbose=0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

@pytest.fixture
def get_n2():
    mol = gto.Mole()
    mol.atom = 'N 0 0 0; N 0 0 1.09'
    mol.basis = '631g'
    mol.verbose=0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

# pylint: disable=redefined-outer-name
def test_nuc_grad(get_h2o):
    mol = get_h2o
    mf = scf.RHF(mol)
    g1 = mf.energy_grad().coords
    mf.kernel()
    g2 = mf.energy_grad().coords
    g0 = mf.Gradients().grad()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6

def test_nuc_grad_deg(get_n2):
    mol = get_n2
    mf = scf.RHF(mol)
    g1 = mf.energy_grad().coords
    mf.kernel()
    g2 = mf.energy_grad().coords
    g0 = mf.Gradients().grad()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6
