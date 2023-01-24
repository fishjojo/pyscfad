import pytest
import numpy
from pyscfad import gto, dft

BOHR = 0.52917721092
disp = 1e-4

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
    mol.basis = '631g'
    mol.build(trace_exp=False, trace_ctr_coeff=False)
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

# pylint: disable=redefined-outer-name
def test_rks_nuc_grad_lda(get_mol):
    mol = get_mol
    mf = dft.RKS(mol)
    mf.xc = 'lda,vwn'
    g1 = mf.energy_grad(mode="rev").coords
    mf.kernel()
    g2 = mf.energy_grad(mode="rev").coords
    g0 = mf.nuc_grad_method().kernel()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6

def test_rks_nuc_grad_gga(get_mol):
    mol = get_mol()
    mf = dft.RKS(mol)
    mf.xc = 'pbe,pbe'
    g1 = mf.energy_grad(mode="rev").coords
    mf.kernel()
    g2 = mf.energy_grad(mode="rev").coords
    g0 = mf.nuc_grad_method().kernel()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6

def test_rks_nuc_grad_hybrid(get_mol):
    mol = get_mol()
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    g1 = mf.energy_grad(mode="rev").coords
    mf.kernel()
    g2 = mf.energy_grad(mode="rev").coords
    g0 = mf.nuc_grad_method().kernel()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6

#FIXME MGGA is broken since pyscf v2.1
def test_rks_nuc_grad_mgga_skip(get_mol, get_mol_p, get_mol_m):
    mol = get_mol
    mf = dft.RKS(mol)
    mf.xc = 'm062x'
    g1 = mf.energy_grad(mode="rev").coords
    mf.kernel()
    g2 = mf.energy_grad(mode="rev").coords
    assert abs(g1 - g2).max() < 2e-6

    molp = get_mol_p
    mfp = dft.RKS(molp)
    mfp.xc = 'm062x'
    ep = mfp.kernel()

    molm = get_mol_m
    mfm = dft.RKS(molm)
    mfm.xc = 'm062x'
    em = mfm.kernel()

    g_fd = (ep-em) / disp * BOHR
    assert abs(g2[1,2] - g_fd) < 3e-6

#FIXME NLC gradient may have bugs, need check
def test_rks_nuc_grad_nlc_skip(get_mol, get_mol_p, get_mol_m):
    mol = get_mol
    mf = dft.RKS(mol)
    mf.xc = 'B97M_V'
    mf.nlc = 'VV10'
    g = mf.energy_grad(mode="rev").coords

    molp = get_mol_p
    mfp = dft.RKS(molp)
    mfp.xc = 'B97M_V'
    mfp.nlc = 'VV10'
    ep = mfp.kernel()

    molm = get_mol_m
    mfm = dft.RKS(molm)
    mfm.xc = 'B97M_V'
    mfm.nlc = 'VV10'
    em = mfm.kernel()

    g_fd = (ep-em) / disp * BOHR
    assert abs(g[1,2] - g_fd) < 2e-4
