import pytest
from pyscfad import gto

@pytest.fixture(scope="module")
def get_h2():
    mol = gto.Mole()
    mol.atom    = 'H 0 0 0; H 0 0 0.74'
    mol.basis   = '631g'
    mol.verbose = 0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

@pytest.fixture(scope="module")
def get_h2o():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = 'sto3g'
    mol.verbose=0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

@pytest.fixture(scope="module")
def get_h2o_plus():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = 'sto3g'
    mol.spin = 1
    mol.charge = 1
    mol.verbose=0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

@pytest.fixture(scope="module")
def get_n2():
    mol = gto.Mole()
    mol.atom = 'N 0 0 0; N 0 0 1.09'
    mol.basis = '631g'
    mol.verbose=0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol
