'''
Raman susceptibility
'''
import numpy
import jax
from pyscfad import gto, scf
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
    def dip_moment(E):
        mf.get_hcore = lambda *args, **kwargs: h1 + jnp.einsum('x,xij->ij', E, ao_dip)
        mf.kernel()
        dip = mf.dip_moment(mol, mf.make_rdm1(), unit='AU', verbose=0)
        return dip
    polar = jax.jacrev(dip_moment)(E)
    return polar

chi = jax.jacrev(polarizability)(mol).coords
print(chi)
