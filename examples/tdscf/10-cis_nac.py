import numpy
import jax
from pyscfad import gto, scf
from pyscfad.tdscf.rhf import CIS, cis_ovlp

# molecular structure
mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 1.1'
mol.basis = 'cc-pvdz'
mol.build(trace_exp=False, trace_ctr_coeff=False)

# HF and CIS calculations
mf = scf.RHF(mol)
mf.kernel()
mytd = CIS(mf)
mytd.nstates = 8
e, xy = mytd.kernel()

# Target excited states I and J (1st and 4th)
stateI, stateJ = 0, 2
# CI coefficients of state I
xi = xy[stateI][0] * numpy.sqrt(2.)
nmo = mf.mo_coeff.shape[-1]
nocc = mol.nelectron // 2

def ovlp(mol1):
    mf1 = scf.RHF(mol1)
    mf1.kernel()
    mytd1 = CIS(mf1)
    mytd1.nstates = 8
    _, xy1 = mytd1.kernel()
    # CI coefficients of state J
    xj = xy1[stateJ][0] * numpy.sqrt(2.)
    # CIS wavefunction overlap
    s = cis_ovlp(mol, mol1, mf.mo_coeff, mf1.mo_coeff,
                 nocc, nocc, nmo, nmo, xi, xj)
    return s

# Only the ket state is differentiated
mol1 = mol.copy()
jac = jax.jacrev(ovlp)(mol1)
print("CIS derivative coupling:")
print(jac.coords)
