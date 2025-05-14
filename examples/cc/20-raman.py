'''
Vibrational frequency
Raman activity

Reference numbers:
vib freq (cm^-1):    2.33669984e+03
Raman act (A^4/amu): 217.844670395618
Depol ratio:         0.556979266301141
'''
import jax
from pyscfad import gto, scf, cc
from pyscfad import numpy as np
from pyscfad.prop.thermo import vib

mol = gto.Mole()
mol.atom = '''B  ,  0.   0.   0.
              H  ,  0.   0.   2.36328'''
mol.basis = 'aug-cc-pvdz'
mol.unit = 'B'
mol.build(trace_exp=False, trace_ctr_coeff=False)

# CCSD energy
def energy(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = cc.RCCSD(mf)
    mycc.kernel()
    return mycc.e_tot

# hessian in cartesian coordinates
hess = jax.jacrev(jax.jacrev(energy))(mol).coords.coords

# CCSD energy with external electric field applied
def apply_E(mol, E):
    field = np.einsum('x,xij->ij', E, mol.intor('int1e_r'))
    mf = scf.RHF(mol)
    h1 = mf.get_hcore() + field
    mf.get_hcore = lambda *args, **kwargs: h1
    mf.kernel()
    mycc = cc.RCCSD(mf)
    mycc.kernel()
    return mycc.e_tot

# zero field
E0 = np.zeros((3))
# Raman tensor in au
chi = -jax.jacrev(jax.jacrev(jax.jacrev(apply_E,1),1),0)(mol, E0).coords

vibration, _, raman = vib.harmonic_analysis(mol, hess, raman_tensor=chi)
print("Vibrational frequency in cm^-1:")
print(vibration['freq_wavenumber'])
print('Raman activity in A^4/amu:')
print(raman['activity'])
print('Depolarization ration:')
print(raman['depolar_ratio'])
