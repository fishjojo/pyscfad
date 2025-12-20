"""RHF energy gradient
"""
import jax
from pyscfad import gto, scf

mol = gto.Mole()
mol.atom = "H 0 0 0; H 0 0 0.74"  # in Angstrom
mol.basis = "631g*"
mol.verbose = 4
mol.build()

def rhf_energy(mol):
    mf = scf.RHF(mol)
    ehf = mf.kernel()
    return ehf

jac = jax.grad(rhf_energy)(mol)
print(f"Nuclear gradient:\n{jac.coords}")
print(f"Gradient w.r.t. basis exponents:\n{jac.exp}")
print(f"Gradient w.r.t. basis contraction coefficients:\n{jac.ctr_coeff}")
