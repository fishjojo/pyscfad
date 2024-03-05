import pytest
import numpy
import jax
from pyscfad import gto, scf, cc

def test_nuc_grad(get_mol):
    mol = get_mol
    def energy(mol):
        mf = scf.RHF(mol)
        mf.kernel()
        mycc = cc.RCCSD(mf)
        mycc.kernel()
        et = mycc.ccsd_t()
        return mycc.e_tot + et
    with jax.disable_jit():
        g1 = jax.grad(energy)(mol).coords
    g0 = numpy.array([[0., 0., -8.60709468e-02],
                      [0., 0.,  8.60709468e-02]])
    assert(abs(g1-g0).max() < 1e-6)
