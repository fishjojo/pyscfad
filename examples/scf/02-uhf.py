"""UHF nuclear gradient
"""
import jax
from pyscfad import gto, scf

#
# 1. open-shell system
#
mol = gto.Mole()
mol.atom = """
    O 0.  0.    0.
    H 0. -0.757 0.587
    H 0.  0.757 0.587
"""
mol.basis = "cc-pvdz"
mol.spin   = 1
mol.charge = 1
mol.verbose = 4
mol.build(trace_exp=False, trace_ctr_coeff=False)

def uhf_energy(mol):
    mf = scf.UHF(mol)
    e = mf.kernel()
    return e, mf

jac, mf = jax.grad(uhf_energy, has_aux=True)(mol)
g1 = jac.coords
g0 = mf.nuc_grad_method().kernel() # Analytic gradient by pyscf
assert abs(g1 - g0).max() < 1e-6

#
# 2. closed-shell system
#
mol.spin   = 0
mol.charge = 0
mol.build(trace_exp=False, trace_ctr_coeff=False)

jac, mf = jax.grad(uhf_energy, has_aux=True)(mol)
g1 = jac.coords
g0 = mf.nuc_grad_method().kernel() # Analytic gradient by pyscf
assert abs(g1 - g0).max() < 1e-6
