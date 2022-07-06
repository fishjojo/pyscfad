'''
IR intensity

vib freq (cm^-1):      1.70028264e+03, 3.84403707e+03, 3.95658888e+03
IR intensity (km/mol): 56.40822073344795
'''
import numpy
import jax
from pyscf.data import elements
from pyscfad import gto, scf, cc
from pyscfad.lib import numpy as np

unit2cm = 5140.4871384933695
unit_kmmol = 974.8801118351438

mol = gto.Mole()
mol.atom =[['O', [0.0000,  0.0000,  0.1212]],
           ['H', [0.0000,  0.7508, -0.4848]],
           ['H', [0.0000, -0.7508, -0.4848]]]
mol.basis = 'cc-pvdz'
mol.build(trace_exp=False, trace_ctr_coeff=False)
natm = mol.natm

# CCSD energy
def energy(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = cc.RCCSD(mf)
    mycc.kernel()
    return mycc.e_tot

hess_cart = jax.jacrev(jax.jacrev(energy))(mol).coords.coords

atom_charges = mol.atom_charges()
mass = numpy.array([elements.MASSES[atom_charges[i]] for i in range(natm)])

# mass weighted hessian
hess_mass = numpy.einsum('ixjy,i,j->ixjy', hess_cart, mass**-.5, mass**-.5)
h = hess_mass.reshape(natm*3, natm*3)
e, c = numpy.linalg.eigh(h)
# normal mode in amu^-.5
norm_mode = numpy.einsum('i,kix->kix', mass**-.5, c.T.reshape(-1,natm,3))

freq = numpy.sign(e) * abs(e)**.5 * unit2cm
print("Vibrational frequency in cm^-1:")
print(freq)

# CCSD dipole moment with external electric field applied
def dipole(mol, E):
    def energy(mol, E):
        mf = scf.RHF(mol)
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        h1 = mf.get_hcore()
        mf.get_hcore = lambda *args, **kwargs: h1 + np.einsum('x,xij->ij', E, ao_dip)
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
E0 = numpy.zeros((3))
# polarizability
alpha = jax.jacrev(dipole)(mol, E0).coords

# norm_mode[6] is bending
a = numpy.einsum('inx,nx->i', alpha, norm_mode[6])
I = numpy.dot(a, a) * unit_kmmol
print(f'IR intensity (km/mol): {I}')
