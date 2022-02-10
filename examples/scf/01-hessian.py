import jax
import pyscf
from pyscfad import gto, scf

"""
Analytic nuclear Hessian for RHF computed by auto-differentiation
"""

mol = gto.Mole()
mol.atom = [
        ['O' , 0. , 0.     , 0],
        ['H' , 0. , -0.757 , 0.587],
        ['H' , 0. ,  0.757 , 0.587]]
mol.basis = '631g'
mol.verbose=5
mol.build(trace_exp=False, trace_ctr_coeff=False)

def ehf(mol):
    mf = scf.RHF(mol)
    e = mf.kernel()
    return e

jac = jax.jacrev(jax.jacrev(ehf))(mol)
print(jac.coords.coords)
