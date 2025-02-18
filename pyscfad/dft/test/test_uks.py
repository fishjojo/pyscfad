from numpy.testing import assert_almost_equal
import jax
from pyscfad import dft
from pyscfad import config_update

def _mf(mol, xc):
    mf = dft.UKS(mol)
    mf.xc = xc
    return mf

def _test_uks_nuc_grad(mol, xc):
    def _eks(mol):
        return _mf(mol, xc).kernel()

    g1 = jax.grad(_eks)(mol).coords

    mf = _mf(mol, xc)
    mf.kernel()
    g0 = mf.nuc_grad_method().kernel()

    assert_almost_equal(g0, g1, decimal=6)

def test_uks_nuc_grad_lda(get_mol):
    _test_uks_nuc_grad(get_mol, "lda,vwn")

def test_uks_nuc_grad_gga(get_mol):
    _test_uks_nuc_grad(get_mol, "pbe,pbe")

def test_uks_nuc_grad_hybrid(get_mol):
    _test_uks_nuc_grad(get_mol, "b3lyp")

def test_uks_nuc_grad_lrc(get_mol):
    _test_uks_nuc_grad(get_mol, "HYB_GGA_XC_LRC_WPBE")

def test_df_uks_nuc_grad(get_mol):
    def _eks(mol):
        return _mf(mol, "pbe,pbe").density_fit().kernel()

    with config_update("pyscfad_scf_implicit_diff", True):
        g = jax.grad(_eks)(get_mol).coords
        # finite difference reference
        assert abs(g[1,2] - -0.007387325326884536) < 1e-6
