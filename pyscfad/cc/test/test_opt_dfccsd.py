import numpy
import jax
from pyscfad import scf
from pyscfad.cc import dfccsd

def test_dfccsdt_nuc_grad(get_opt_mol):
    mol = get_opt_mol
    def energy(mol):
        mf = scf.RHF(mol).density_fit()
        mf.kernel()

        mycc = dfccsd.RCCSD(mf)
        eris = mycc.ao2mo()
        mycc.kernel(eris=eris)
        et = mycc.ccsd_t(eris=eris)
        return mycc.e_tot + et

    e, jac = jax.value_and_grad(energy)(mol)

    e0 = -100.10156178822595
    assert abs(e - e0) < 1e-6

    g0 = numpy.array([[0., 0., -0.0860735932],
                      [0., 0.,  0.0860735932]])
    assert abs(jac.coords-g0).max() < 1e-6
