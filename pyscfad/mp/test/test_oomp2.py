import pytest
import numpy
import jax
from scipy.optimize import minimize
from pyscfad import util
from pyscfad.lib import numpy as np
from pyscfad.tools import rotate_mo1
from pyscfad import gto, scf, mp
from pyscfad import config
config.update('pyscfad_scf_implicit_diff', True)
#config.update('pyscfad_moleintor_opt', True)

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom ='H 0 0 0; F 0 0 1.1'
    mol.basis = '631g'
    mol.build()
    return mol

@util.pytree_node(['_scf', 'x'], num_args=1)
class OOMP2(mp.MP2):
    def __init__(self, mf, x=None, **kwargs):
        mp.MP2.__init__(self, mf)
        self.x = x
        self.__dict__.update(kwargs)
        if self.x is None:
            nao = self.mol.nao
            assert nao == self.nmo
            size = nao*(nao+1)//2
            self.x = np.zeros([size,])
        self.mo_coeff = rotate_mo1(self._scf.mo_coeff, self.x) 
        self._scf.converged = False

def func(x0, mf):
    def energy(x0, mf):
        mymp = OOMP2(mf, x=np.asarray(x0))
        mymp.kernel()
        return mymp.e_tot

    def grad(x0, mf):
        f, g = jax.value_and_grad(energy)(x0, mf)
        return f, g

    f, g = grad(x0, mf)
    return (numpy.array(f), numpy.array(g))

def test_oomp2_energy(get_mol):
    mol = get_mol
    mf = scf.RHF(mol)
    mf.kernel()
    nao = mol.nao
    size = nao*(nao+1)//2
    x0 = numpy.zeros([size,])
    options = {"gtol":1e-5}
    res = minimize(func, x0, args=(mf,), jac=True, method="BFGS", options = options)
    e = func(res.x, mf)[0]
    assert abs(e - -100.0986986661949) < 1e-8
