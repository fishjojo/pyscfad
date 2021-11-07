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

mf           = dft.UKS(mol)
mf.xc        = 'LDA'
mf.max_cycle = 100
mf.kernel()
jac = mf.energy_grad()
print(jac.coords)
