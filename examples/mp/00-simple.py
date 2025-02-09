"""
Analytic nuclear gradient for MP2 computed by auto-differentiation
"""
import jax
import pyscf
from pyscfad import gto, scf, mp
from pyscfad import config

# implicit differentiation of SCF iterations
config.update('pyscfad_scf_implicit_diff', True)

mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'
mol.basis = '631g'
mol.verbose=4
mol.build()

def mp2(mol, dm0=None):
    mf = scf.RHF(mol)
    mf.kernel(dm0)
    mymp = mp.MP2(mf)
    mymp.kernel()
    return mymp.e_tot

jac = jax.grad(mp2)(mol)
print(f'Nuclaer gradient:\n{jac.coords}')
print(f'Gradient wrt basis exponents:\n{jac.exp}')
print(f'Gradient wrt basis contraction coefficients:\n{jac.ctr_coeff}')
