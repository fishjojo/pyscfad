from pyscfad import gto, scf

"""
Analytic nuclear gradient for RHF computed by auto-differentiation
"""

mol = gto.Mole()
mol.atom    = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
mol.basis   = '631g'
mol.verbose = 5
mol.build()

from pyscfad import config
#config.update('pyscfad_scf_implicit_diff', True)

def energy(mol):
    mf = scf.RHF(mol)
    ehf = mf.kernel()
    return ehf
import jax
jac = jax.grad(energy)(mol)
#jac = mf.energy_grad()
print(f'Nuclaer gradient:\n{jac.coords}')
print(f'Gradient wrt basis exponents:\n{jac.exp}')
print(f'Gradient wrt basis contraction coefficients:\n{jac.ctr_coeff}')
