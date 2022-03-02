import pytest
import numpy
import jax
from pyscfad import gto
from pyscfad import scf
from pyscfad import cc

@pytest.fixture
def get_hf():
    mol = gto.Mole()
    mol.atom = 'H 0. 0. 0.; F 0. 0. 1.1'
    mol.basis = '631g'
    mol.verbose = 0
    mol.incore_anyway = True
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

def test_nuc_grad(get_hf):
    mol = get_hf
    def energy(mol):
        mf = scf.RHF(mol)
        mf.kernel()
        mycc = cc.RCCSD(mf)
        mycc.kernel()
        return mycc.e_tot
    g1 = jax.jacrev(energy)(mol).coords
    g0 = numpy.array([[0., 0., -8.73564765e-02],
                      [0., 0.,  8.73564765e-02]])
    assert(abs(g1-g0).max() < 1e-6)
