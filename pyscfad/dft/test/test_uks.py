import jax
from pyscfad import gto, dft
from pyscfad import config_update

BOHR = 0.52917721092
disp = 1e-4

def test_uks_nuc_grad_lda(get_mol):
    mol = get_mol
    mf = dft.UKS(mol)
    mf.xc = 'lda,vwn'
    g1 = mf.energy_grad(mode="rev").coords
    mf.kernel()
    g2 = mf.energy_grad(mode="rev").coords
    g0 = mf.nuc_grad_method().kernel()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6

def test_uks_nuc_grad_gga(get_mol):
    mol = get_mol()
    mf = dft.UKS(mol)
    mf.xc = 'pbe,pbe'
    g1 = mf.energy_grad(mode="rev").coords
    mf.kernel()
    g2 = mf.energy_grad(mode="rev").coords
    g0 = mf.nuc_grad_method().kernel()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6

def test_uks_nuc_grad_hybrid(get_mol):
    mol = get_mol()
    mf = dft.UKS(mol)
    mf.xc = 'b3lyp'
    g1 = mf.energy_grad(mode="rev").coords
    mf.kernel()
    g2 = mf.energy_grad(mode="rev").coords
    g0 = mf.nuc_grad_method().kernel()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6

def test_uks_nuc_grad_lrc(get_mol):
    mol = get_mol()
    mf = dft.UKS(mol)
    mf.xc = 'HYB_GGA_XC_LRC_WPBE'
    g1 = mf.energy_grad(mode="rev").coords
    mf.kernel()
    g2 = mf.energy_grad(mode="rev").coords
    g0 = mf.nuc_grad_method().kernel()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6

def test_df_uks_nuc_grad(get_mol):
    with config_update('pyscfad_scf_implicit_diff', True):
        mol = get_mol
        def energy(mol):
            mf = dft.UKS(mol, xc='PBE,PBE').density_fit()
            return mf.kernel()
        g = jax.grad(energy)(mol).coords
        # finite difference reference
        assert abs(g[1,2] - -0.007387325326884536) < 1e-6
