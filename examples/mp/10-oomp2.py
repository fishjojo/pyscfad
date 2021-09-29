import jax
jax.config.update("jax_enable_x64", True)
import numpy
import jax.scipy as scipy
from pyscfad import lib
from pyscfad.lib import numpy as jnp
from pyscfad.lib import ops
from pyscfad.lib.numpy_helper import unpack_triu
from pyscfad.tools import rotate_mo1
from pyscfad import gto, scf
from pyscfad import mp

@lib.dataclass
class OOMP2(mp.MP2):
    x: jnp.array = lib.field(pytree_node=True, default=None)

    def __post_init__(self):
        mp.MP2.__post_init__(self)
        if self.x is None:
            nao = self.mol.nao
            assert nao == self.nmo
            size = nao*(nao+1)//2
            self.x = jnp.zeros([size,])
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
        mymp = OOMP2(mf, x=jnp.asarray(x0))
        mymp.kernel()
        return mymp.e_tot

    def grad(x0, mf):
        f, g = jax.value_and_grad(energy)(x0, mf)
        return f, g

    f, g = grad(x0, mf)
    return (numpy.array(f), numpy.array(g))

from scipy.optimize import minimize
nao = mol.nao
size = nao*(nao+1)//2
x0 = numpy.zeros([size,])
options = {"gtol":1e-6}
res = minimize(func, x0, args=(mf,), jac=True, method="BFGS", options = options)
e = func(res.x, mf)[0]
print(e)
