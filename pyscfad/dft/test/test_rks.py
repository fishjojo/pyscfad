import pytest
import numpy
from pyscfad import gto, dft

BOHR = 0.52917721092

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
    mol.basis = '631g'
    mol.build(trace_coords=True)
    return mol

@pytest.fixture
def get_mol_p():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74005'  # in Angstrom
    mol.basis = '631g'
    mol.build()
    return mol

@pytest.fixture
def get_mol_m():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.73995'  # in Angstrom
    mol.basis = '631g'
    mol.build()
    return mol

def test_rks_nuc_grad(get_mol):
    mol = get_mol
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    g = mf.energy_grad(mode="rev").coords
    g0 = numpy.array([[0, 0, 2.24114270e-03],
                      [0, 0, -2.24114270e-03]])
    assert abs(g-g0).max() < 1e-10

def test_rks_nuc_grad_mgga(get_mol, get_mol_p, get_mol_m):
    # meta-GGA
    mol = get_mol
    mf = dft.RKS(mol)
    mf.xc = 'm062x'
    g = mf.energy_grad(mode="rev").coords

    molp = get_mol_p
    mfp = dft.RKS(molp)
    mfp.xc = 'm062x'
    ep = mfp.kernel()

    molm = get_mol_m
    mfm = dft.RKS(molm)
    mfm.xc = 'm062x'
    em = mfm.kernel()

    g_fd = (ep-em) / 1e-4 * BOHR
    assert abs(g[1,2] - g_fd) < 3e-6

def test_rks_nuc_grad_nlc(get_mol):
    mol = get_mol
    mf = dft.RKS(mol)
    mf.xc = 'B97M_V'
    mf.nlc = 'VV10'
    g = mf.energy_grad(mode="rev").coords
    assert abs(g[1,2] - 2.68791294e-03) < 1e-9
