import pytest
import numpy
import jax
import pyscf
from pyscfad import gto, scf, mp

@pytest.fixture
def get_mol0():
    mol = pyscf.M(
        atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587',
        basis = 'sto3g',
        verbose=0,
    )
    return mol

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = 'sto3g'
    mol.verbose=0
    mol.build()
    return mol

def test_nuc_grad(get_mol0, get_mol):
    mol = get_mol
    def mp2(mol):
        mf = scf.RHF(mol)
        mf.kernel()
        mymp = mp.MP2(mf)
        mymp.kernel()
        return mymp.e_tot
    g = jax.grad(mp2)(mol).coords
    g0 = numpy.asarray([[0, 0, 9.49104413e-02],
                        [0, 3.79148621e-02, -4.74552207e-02],
                        [0, -3.79148621e-02, -4.74552207e-02]])
    assert abs(g-g0).max() < 2e-6
