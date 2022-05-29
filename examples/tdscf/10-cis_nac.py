import jax
from jax import numpy as jnp
from pyscfad import gto, scf, tdscf

mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 1.1'
mol.basis = 'ccpvtz'
mol.verbose = 4
mol.build(trace_exp=False, trace_ctr_coeff=False)

mf = scf.RHF(mol)
mf.kernel()
mytd = tdscf.rhf.CIS(mf)
mytd.nstates = 3
e, xy = mytd.kernel()

i, j = 0, 2
xi = xy[i][0] * jnp.sqrt(2.)
xj = xy[j][0] * jnp.sqrt(2.)

# Using Hellman-Feynman formalism.
# The amplitude is closed over, so there is no tracing through the Davidson iteration.
def hellman(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    mytd = tdscf.rhf.CIS(mf)
    mytd.nstates = 3

    vind, _ = mytd.gen_vind(mytd._scf)
    e = jnp.dot(xi.ravel(), vind(xj).ravel())
    return e

nac = jax.grad(hellman)(mol)
print(nac.coords / (e[j]-e[i]))
