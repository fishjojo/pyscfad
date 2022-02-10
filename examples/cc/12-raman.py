'''
Raman susceptibility
'''
import numpy
import jax
from pyscfad import gto, scf, cc
from pyscfad.lib import numpy as jnp

mol = gto.Mole()
mol.atom = '''H  ,  0.   0.   0.
              F  ,  0.   0.   .917'''
mol.basis = '631g'
mol.build(trace_exp=False, trace_ctr_coeff=False)

def polarizability(mol):
    mf = scf.RHF(mol)
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    h1 = mf.get_hcore()
    E = numpy.zeros((3))
    def energy(E):
        mf.get_hcore = lambda *args, **kwargs: h1 + jnp.einsum('x,xij->ij', E, ao_dip)
        mf.kernel()
        mycc = cc.RCCSD(mf)
        e_tot = mycc.kernel()[0]
        return e_tot
    polar = jax.jacfwd(jax.jacrev(energy))(E)
    return polar

import time
t0 = time.time()
chi = jax.jacfwd(polarizability)(mol).coords
print(chi)
print(time.time() - t0)
