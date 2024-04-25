'''
IR intensity

vib freq (cm^-1):      [1700.28481202 3844.03982493 3956.57239063]
IR intensity (km/mol): [56.40777805  4.46776132 22.64275908]
'''
import jax
from pyscfad import gto, scf, cc
from pyscfad.lib import numpy as np
from pyscfad.prop.thermo import vib

mol = gto.Mole()
mol.atom =[['O', [0.0000,  0.0000,  0.1212]],
           ['H', [0.0000,  0.7508, -0.4848]],
           ['H', [0.0000, -0.7508, -0.4848]]]
mol.basis = 'cc-pvdz'
mol.build(trace_exp=False, trace_ctr_coeff=False)

# CCSD energy
def energy(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = cc.RCCSD(mf)
    mycc.kernel()
    return mycc.e_tot

hess = jax.jacrev(jax.jacrev(energy))(mol).coords.coords

# CCSD dipole moment with external electric field applied
def dipole(mol, E):
    def energy(mol, E):
        field = np.einsum('x,xij->ij', E, mol.intor('int1e_r'))
        mf = scf.RHF(mol)
        h1 = mf.get_hcore() + field
        mf.get_hcore = lambda *args, **kwargs: h1
        mf.kernel()
        mycc = cc.RCCSD(mf)
        mycc.kernel()
        return mycc.e_tot

    dip_elec = -jax.jacrev(energy, 1)(mol, E)
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    nucl_dip = np.einsum('i,ix->x', charges, coords)
    return nucl_dip + dip_elec

# zero field
E0 = np.zeros((3))
# IR tensor
ir_tensor = jax.jacrev(dipole)(mol, E0).coords

vibration, ir, _ = vib.harmonic_analysis(mol, hess, ir_tensor)
print("Vibrational frequency in cm^-1:")
print(vibration['freq_wavenumber'])
print('IR intensity (km/mol):')
print(ir['intensity'])
