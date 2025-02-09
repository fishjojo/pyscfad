'''
IR intensity

Vibrational frequency in cm^-1:
[1775.68298016 4113.24151671 4211.55679665]
IR intensity (km/mol):
[80.68436347 21.17123945 60.46711041]
'''
import jax
from pyscfad import gto, scf, cc
from pyscfad import numpy as np
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

def dipole(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    return mf.dip_moment(mol, mf.make_rdm1(), unit='AU', verbose=0)

# IR tensor
ir_tensor = jax.jacrev(dipole)(mol).coords

vibration, ir, _ = vib.harmonic_analysis(mol, hess, ir_tensor)
print("Vibrational frequency in cm^-1:")
print(vibration['freq_wavenumber'])
print('IR intensity (km/mol):')
print(ir['intensity'])
