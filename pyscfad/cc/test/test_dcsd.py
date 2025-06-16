import numpy
import jax
from pyscfad import scf
from pyscfad.cc import dfdcsd

def test_df_nuc_grad(get_mol):
    mol = get_mol
    def energy(mol):
        mf = scf.RHF(mol).density_fit()
        mf.kernel()
        mycc = dfdcsd.RDCSD(mf)
        mycc.kernel()
        return mycc.e_tot
    g1 = jax.grad(energy)(mol).coords
    # finite difference
    g0 = numpy.array([[0., 0., -0.08500490828],
                      [0., 0.,  0.08500490828]])
    assert(abs(g1-g0).max() < 1e-6)
