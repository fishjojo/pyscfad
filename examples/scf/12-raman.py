'''
Raman activity

Vibrational frequency in cm^-1:
[1775.68298016 4113.24151671 4211.55679665]
Raman activity in A^4/amu:
[ 4.78998643 68.87902111 34.78821604]
Depolarization ration:
[0.52686036 0.17011081 0.75      ]
'''
import jax
from pyscfad import gto, scf
from pyscfad import numpy as np
from pyscfad.prop.polarizability.rhf import Polarizability
from pyscfad.prop.thermo import vib

mol = gto.Mole()
mol.atom =[['O', [0.0000,  0.0000,  0.1157]],
           ['H', [0.0000,  0.7488, -0.4629]],
           ['H', [0.0000, -0.7488, -0.4629]]]
mol.basis = 'cc-pvdz'
mol.build(trace_exp=False, trace_ctr_coeff=False)

def energy(mol):
    mf = scf.RHF(mol)
    e_tot = mf.kernel()
    return e_tot

hess = jax.jacrev(jax.jacrev(energy))(mol).coords.coords

def apply_E(mol, E):
    field = np.einsum('x,xij->ij', E, mol.intor('int1e_r'))
    mf = scf.RHF(mol)
    h1 = mf.get_hcore() + field
    mf.get_hcore = lambda *args, **kwargs: h1
    e_tot = mf.kernel()
    return e_tot

def polarizability(mol, freq=0.0):
    mf = scf.RHF(mol)
    mf.kernel()
    alpha = Polarizability(mf).polarizability_with_freq(freq=freq)
    return alpha

# equivalent ways to compute chi
# method2 allows frequency dependent polarizability
method = 2
if method == 1:
    E0 = np.zeros((3))
    chi = -jax.jacrev(jax.jacrev(jax.jacrev(apply_E,1),1),0)(mol, E0).coords
else:
    chi = jax.jacrev(polarizability)(mol, freq=0.0).coords

vibration, _, raman = vib.harmonic_analysis(mol, hess, raman_tensor=chi)
print("Vibrational frequency in cm^-1:")
print(vibration['freq_wavenumber'])
print('Raman activity in A^4/amu:')
print(raman['activity'])
print('Depolarization ration:')
print(raman['depolar_ratio'])
