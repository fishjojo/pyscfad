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
        return mycc.e_tot
    with jax.disable_jit():
        g1 = jax.jacrev(energy)(mol).coords
    g0 = numpy.array([[0., 0., -0.0873564848],
                      [0., 0.,  0.0873564848]])
    assert(abs(g1-g0).max() < 1e-6)
