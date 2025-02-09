'''
Bohrn effective charge tensor
'''
import numpy
import jax
from pyscfad import gto, scf
from pyscfad import numpy as np

mol = gto.Mole()
mol.atom = '''H  ,  0.   0.   0.
              F  ,  0.   0.   .917'''
mol.basis = '631g'
mol.build(trace_exp=False, trace_ctr_coeff=False)

def dip_moment(mol):
    mf = scf.RHF(mol)
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    h1 = mf.get_hcore()
    E = numpy.zeros((3))
    mf.get_hcore = lambda *args, **kwargs: h1 + np.einsum('x,xij->ij', E, ao_dip)
    mf.kernel()
    return mf.dip_moment(mol, mf.make_rdm1(), unit='AU', verbose=0)

z = jax.jacrev(dip_moment)(mol).coords
print(z)
