import pytest
from jax import jacrev
from pyscf.lib import fp
from pyscfad import gto, dft

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
    mol.basis = '631g'
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

def energy(mol, xc):
    mf = dft.RKS(mol)
    mf.xc = xc
    e = mf.kernel()
    return e

def test_rks_nuc_hess_lda(get_mol):
    mol = get_mol
    hess = jacrev(jacrev(energy))(mol, 'lda,vwn').coords.coords
    assert abs(fp(hess) - -0.5301815984221748) < 1e-6

def test_rks_nuc_hess_gga(get_mol):
    mol = get_mol
    hess = jacrev(jacrev(energy))(mol, 'pbe, pbe').coords.coords
    assert abs(fp(hess) - -0.5146764054396936) < 1e-6

def test_rks_nuc_hess_gga_hybrid(get_mol):
    mol = get_mol
    hess = jacrev(jacrev(energy))(mol, 'b3lyp').coords.coords
    assert abs(fp(hess) - -0.5114248632559669) < 1e-6
