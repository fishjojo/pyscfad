from scipy.optimize import minimize
from jax import value_and_grad
from pyscf import numpy as np
from pyscfad import util
from pyscfad.tools import rotate_mo1
from pyscfad import gto, scf, mp

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

mol = gto.Mole()
mol.atom ='H 0 0 0; F 0 0 1.1'
mol.basis = 'ccpvdz'
mol.build()
mf = scf.RHF(mol)
mf.kernel()

def func(x0, mf):
    def energy(x0, mf):
        mymp = OOMP2(mf, x=np.asarray(x0))
        mymp.kernel()
        return mymp.e_tot

    def grad(x0, mf):
        f, g = value_and_grad(energy)(x0, mf)
        return f, g

    f, g = grad(x0, mf)
    return (np.array(f), np.array(g))

nao = mol.nao
size = nao*(nao+1)//2
x0 = np.zeros([size,])
options = {"gtol":1e-5}
res = minimize(func, x0, args=(mf,), jac=True, method="BFGS", options = options)
e = func(res.x, mf)[0]
print(e)
