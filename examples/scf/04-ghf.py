"""GHF energy gradient
"""
import jax
from pyscfad import gto, scf

mol = gto.Mole()
mol.atom = """
    O 0.  0.    0.
    H 0. -2.757 2.587
    H 0.  2.757 2.587
"""
mol.basis = "ccpvdz"
mol.charge = 1
mol.spin = 1
mol.build()

def ghf_energy(mol, dm=None):
    mf = scf.GHF(mol)
    ehf = mf.kernel(dm0=dm)
    return ehf

#
# 1. real GHF
#
e, jac = jax.value_and_grad(ghf_energy)(mol)
print(f"Nuclear gradient:\n{jac.coords}")
print(f"Gradient w.r.t. basis exponents:\n{jac.exp}")
print(f"Gradient w.r.t. basis contraction coefficients:\n{jac.ctr_coeff}")

#
# 2. complex GHF
#
dm = scf.GHF(mol).get_init_guess() + 0j
dm = dm.at[0,:].add(0.05j)
dm = dm.at[:,0].add(-0.05j)

e, jac = jax.value_and_grad(ghf_energy)(mol, dm)
print(f"Nuclear gradient:\n{jac.coords}")
print(f"Gradient w.r.t. basis exponents:\n{jac.exp}")
print(f"Gradient w.r.t. basis contraction coefficients:\n{jac.ctr_coeff}")
