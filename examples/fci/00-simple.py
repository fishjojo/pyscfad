import jax
from pyscfad import gto, scf, fci

# molecular structure
mol = gto.Mole()
mol.atom = 'H 0 0 0; H  0 0 1.1'
mol.basis = 'ccpvdz'
mol.verbose = 4
mol.build()

def fci_energy(mol, nroots=1):
    mf = scf.RHF(mol)
    mf.kernel()
    e, fcivec = fci.solve_fci(mf, nroots=nroots)
    return e

jac = jax.jacrev(fci_energy)(mol)
print(f'Nuclaer gradient:\n{jac.coords}')
print(f'Gradient wrt basis exponents:\n{jac.exp}')
print(f'Gradient wrt basis contraction coefficients:\n{jac.ctr_coeff}')
