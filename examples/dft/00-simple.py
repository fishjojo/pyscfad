"""RKS nuclear gradient
"""
import jax
from pyscfad import gto, dft

mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'
mol.basis = '631G*'
mol.verbose = 4
mol.build()

energy_fn = lambda mol: dft.RKS(mol, xc='b3lyp').kernel() 
jac = jax.grad(energy_fn)(mol)
print(f'Nuclaer gradient:\n{jac.coords}')
print(f'Gradient w.r.t. basis exponents:\n{jac.exp}')
print(f'Gradient w.r.t. basis contraction coefficients:\n{jac.ctr_coeff}')
