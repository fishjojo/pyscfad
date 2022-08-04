import jax
from pyscfad import gto, scf, fci

# molecular structure
mol = gto.Mole()
mol.atom = 'H 0 0 0; H  0 0 1.1'
mol.basis = 'ccpvdz'
mol.build()

# HF and FCI calculation
nroots = 8
mf = scf.RHF(mol)
mf.kernel()
e, fcivec = fci.solve_fci(mf, nroots=nroots)
print(e)

nelec = mol.nelectron
norb = mf.mo_coeff.shape[-1]
stateI, stateJ = 2, 7
def ovlp(mol1):
    mf1 = scf.RHF(mol1)
    mf1.kernel()
    e1, fcivec1 = fci.solve_fci(mf1, nroots=nroots)
    # wavefunction overlap
    s = fci.fci_ovlp(mol, mol1, fcivec[stateI], fcivec1[stateJ],
                     norb, norb, nelec, nelec, mf.mo_coeff, mf1.mo_coeff)
    return s

# Only the ket state is differentiated
mol1 = mol.copy()
jac = jax.jacrev(ovlp)(mol1)
print("FCI derivative coupling:")
print(jac.coords)
