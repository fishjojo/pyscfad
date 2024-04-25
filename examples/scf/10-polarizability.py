import numpy
import jax
from pyscfad import gto, scf
from pyscfad.lib import numpy as jnp

mol = gto.Mole()
mol.atom = '''H  ,  0.   0.   0.
              F  ,  0.   0.   .917'''
mol.basis = '631g'
mol.build()

mf = scf.RHF(mol)
mf.kernel()
ao_dip = mol.intor_symmetric('int1e_r', comp=3)
h1 = mf.get_hcore()
    
def apply_E(E):
    mf.get_hcore = lambda *args, **kwargs: h1 + jnp.einsum('x,xij->ij', E, ao_dip)
    mf.kernel()
    return mf.dip_moment(mol, mf.make_rdm1(), unit='AU', verbose=0)

E0 = numpy.zeros((3))
polar = jax.jacfwd(apply_E)(E0)
print(polar)

def apply_E1(E):
    mf.get_hcore = lambda *args, **kwargs: h1 + jnp.einsum('x,xij->ij', E, ao_dip)
    return mf.kernel()

polar = -jax.hessian(apply_E1)(E0)
print(polar)

# finite difference polarizability
e1 = apply_E([ 0.0001, 0, 0])
e2 = apply_E([-0.0001, 0, 0])
print((e1 - e2) / 0.0002)

e1 = apply_E([0, 0.0001, 0])
e2 = apply_E([0,-0.0001, 0])
print((e1 - e2) / 0.0002)

e1 = apply_E([0, 0, 0.0001])
e2 = apply_E([0, 0,-0.0001])
print((e1 - e2) / 0.0002)

# hyper-polarizability
hpolar = jax.jacfwd(jax.jacfwd(apply_E))(E0)
print(hpolar)

hpolar = -jax.jacfwd(jax.hessian(apply_E1))(E0)
print(hpolar)
