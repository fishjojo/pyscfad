import pytest
import pyscf
from pyscfad import gto, scf

@pytest.fixture
def get_h2o():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = 'sto3g'
    mol.spin = 1
    mol.charge = 1
    mol.verbose=0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

# pylint: disable=redefined-outer-name
def test_nuc_grad(get_h2o):
    mol = get_h2o
    mf = scf.UHF(mol)
    g1 = mf.energy_grad().coords
    mf.kernel()
    g2 = mf.energy_grad().coords
    g0 = mf.Gradients().grad()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6
