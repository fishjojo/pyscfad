"""ROHF nuclear gradient

Reference results from PySCF:
converged SCF energy = -75.578154312784
--------------- ROHF gradients ---------------
         x                y                z
0 O     0.0000000000    -0.0000000000    -0.0023904882
1 H     0.0000000000    -0.0432752607     0.0011952441
2 H    -0.0000000000     0.0432752607     0.0011952441
----------------------------------------------
"""
import jax
from pyscfad import gto, scf

mol = gto.Mole()
mol.atom = """
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161
"""
mol.basis = "631g"
mol.charge = 1
mol.spin = 1  # = 2S = spin_up - spin_down
mol.verbose = 4
mol.build()

def rohf_energy(mol):
    mf = scf.ROHF(mol)
    return mf.kernel()

jac = jax.grad(rohf_energy)(mol)
print(f'Nuclear gradient:\n{jac.coords}')
print(f'Gradient wrt basis exponents:\n{jac.exp}')
print(f'Gradient wrt basis contraction coefficients:\n{jac.ctr_coeff}')
