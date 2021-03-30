import pyscf
from pyscfad import gto, scf

"""
Analytic nuclear gradient for RHF computed by auto-differentiation
"""

mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
mol.basis = '631g'
mol.verbose=5
mol.build()

mf = scf.RHF(mol)
g = mf.nuc_grad_ad()
print(g)
