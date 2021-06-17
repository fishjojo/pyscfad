import jax
jax.config.update("jax_enable_x64", True)
import numpy
import jax.scipy as scipy
from pyscfad import lib
from pyscfad.lib import numpy as jnp
from pyscfad.lib import ops
from pyscfad import gto, scf
from pyscfad import mp

def unpack_triu(x, n, hermi=0):
    R = jnp.zeros([n,n])
    idx = numpy.triu_indices(n)
    R = ops.index_update(R, idx, x)
    if hermi == 0:
        return R
    elif hermi == 1:
        R = R + R.conj().T
        R = ops.index_mul(R, numpy.diag_indices(n), 0.5)
        return R
    elif hermi == 2:
        return R - R.conj().T
    else:
        raise KeyError

def update_rotate_matrix(dx, n, u0=1):
    dr = unpack_triu(dx, n, hermi=2)
    u = jnp.dot(u0, scipy.linalg.expm(dr))
    return u

def rotate_mo(mo_coeff, u):
    mo = jnp.dot(mo_coeff, u)
    return mo

def rotate_mo1(mo_coeff, x):
    nao = mo_coeff.shape[0]
    u = update_rotate_matrix(x, nao)
    mo_coeff1 = rotate_mo(mo_coeff, u)
    return mo_coeff1

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
print("OOMP2 energy: ", e)
