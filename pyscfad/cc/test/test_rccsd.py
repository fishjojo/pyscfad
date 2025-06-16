import numpy
import jax
from pyscfad import scf, cc

def test_nuc_grad(get_mol):
    mol = get_mol
    def energy(mol):
        mf = scf.RHF(mol)
        mf.kernel()
        mycc = cc.RCCSD(mf)
        mycc.kernel()
        return mycc.e_tot
    g1 = jax.grad(energy)(mol).coords
    g0 = numpy.array([[0., 0., -0.0873564848],
                      [0., 0.,  0.0873564848]])
    assert(abs(g1-g0).max() < 1e-6)

def test_df_nuc_grad(get_mol):
    mol = get_mol
    def energy(mol):
        mf = scf.RHF(mol).density_fit()
        mf.kernel()
        mycc = cc.dfccsd.RCCSD(mf)
        mycc.kernel()
        return mycc.e_tot
    g1 = jax.grad(energy)(mol).coords
    # finite difference
    g0 = numpy.array([[0., 0., -0.0873569023],
                      [0., 0.,  0.0873569023]])
    assert(abs(g1-g0).max() < 5e-6)
