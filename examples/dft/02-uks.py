import pyscf
from pyscfad import gto, dft

"""
Analytic nuclear gradient for RKS computed by auto-differentiation
"""

mol = gto.Mole()
mol.atom = '''
  O   -0.0000000   -0.0184041    0.0000000
  H    0.0000000    0.5383516   -0.7830365
  H   -0.0000000    0.5383516    0.7830365
'''
mol.basis   = 'sto-3g'
mol.verbose = 3
mol.build()

mf    = dft.RKS(mol)
mf.xc = 'LDA'
mf.kernel()
jac = mf.energy_grad()
g1  = jac.coords

print("g1 = \n", g1)

grad = mf.nuc_grad_method()
g2  = grad.kernel()

print("g2 = \n", g2)

assert abs(g1-g2).max() < 1e-6