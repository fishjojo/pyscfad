import pyscf
from pyscfad import gto, scf

"""
Analytic nuclear gradient for RHF computed by auto-differentiation
"""

mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
mol.basis = '631g'
mol.verbose=5
mol.build(trace_coords = True,
          trace_ctr_coeff = True,
          trace_exp = True)

mf = scf.RHF(mol)
mf.kernel()
jac = mf.mol_grad_ad()
print(jac.coords)
print(jac.ctr_coeff)
print(jac.exp)
