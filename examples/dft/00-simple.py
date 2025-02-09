from pyscfad import gto, dft

"""
Analytic nuclear gradient for RKS computed by auto-differentiation
"""

mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'
mol.basis = '631g'
mol.verbose=5
mol.build()

# LDA
mf    = dft.RKS(mol)
mf.xc = "LDA"
mf.kernel()

jac = mf.energy_grad()
print(f'Nuclaer gradient:\n{jac.coords}')
print(f'Gradient wrt basis exponents:\n{jac.exp}')
print(f'Gradient wrt basis contraction coefficients:\n{jac.ctr_coeff}')
