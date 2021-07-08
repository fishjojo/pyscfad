import jax
import numpy
from pyscfad import gto
from pyscfad.lib import numpy as jnp

mol = gto.Mole()
mol.atom = 'H 0. 0. 0.0; Li 0. 0. 0.74'
mol.basis = 'sto3g'
mol.build()

def func(mol, intor):
    return mol.intor(intor)

jac = jax.hessian(func)(mol, "int1e_nuc")
print(jac.coords.coords.shape)
#print(jac.coords.exp)
#print(jac.coords.exp)
