import jax
import pyscf
from pyscfad import gto, scf, mp

"""
Analytic nuclear gradient for MP2 computed by auto-differentiation
"""

mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
mol.basis = '631g'
mol.verbose=5
mol.build()

def mp2(mol, dm0=None):
    mf = scf.RHF(mol)
    mf.kernel(dm0)
    mymp = mp.MP2(mf)
    mymp.kernel()
    return mymp.e_tot

jac = jax.grad(mp2)(mol)
print(jac.coords)
print(jac.exp)
print(jac.ctr_coeff)
