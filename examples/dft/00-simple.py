import pyscf
from pyscfad import gto, dft

"""
Analytic nuclear gradient for RKS computed by auto-differentiation
"""

mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
mol.basis = '631g'
mol.verbose=5
mol.build(trace_coords=True)

mf = dft.RKS(mol)
mf.xc = 'b3lyp'
g = mf.nuc_grad_ad(mode="fwd")
print(g)
