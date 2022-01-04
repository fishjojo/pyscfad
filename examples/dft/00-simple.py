import pyscf
from pyscfad import gto, dft

"""
Analytic nuclear gradient for RKS computed by auto-differentiation
"""

mol      = gto.Mole()
mol.atom = '''
  O   -0.0000000   -0.0184041    0.0000000
  H    0.0000000    0.5383516   -0.7830365
  H   -0.0000000    0.5383516    0.7830365
'''
mol.basis   = 'cc-pvdz'
mol.verbose = 0
mol.build()

mf = dft.RKS(mol)
mf.xc = "b3lyp"
mf.kernel()
jac = mf.energy_grad()
g1  = jac.coords

grad = mf.nuc_grad_method()
g2   = grad.kernel()

assert abs(g1-g2).max() < 1-6
