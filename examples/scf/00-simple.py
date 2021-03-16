import pyscf
from pyscfad import gto, scf

"""
Analytic nuclear gradient for RHF computed by auto-differentiation
"""

mol0 = pyscf.M(
    atom = 'H 0 0 0; H 0 0 0.74',  # in Angstrom
    basis = '631g',
    verbose=0,
)
coords = mol0.atom_coords()
mol = gto.Mole(mol0, coords)

mf = scf.RHF(mol)
mf.kernel()
g = mf.nuc_grad_ad()
print(g)
