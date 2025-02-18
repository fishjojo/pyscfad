import pytest
import jax
from pyscfad import gto

@pytest.fixture(autouse=True)
def clear_cache():
    yield
    jax.clear_caches()

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
    mol.basis = '631g'
    mol.max_memory = 7000
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    yield mol

@pytest.fixture
def get_mol_p():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74005'  # in Angstrom
    mol.basis = '631g'
    mol.build()
    yield mol

@pytest.fixture
def get_mol_m():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.73995'  # in Angstrom
    mol.basis = '631g'
    mol.build()
    yield mol
