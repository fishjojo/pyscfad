'''
Vibrational frequency
Raman activity

Reference numbers:
vib freq (cm^-1):    2.33669984e+03
Raman act (A^4/amu): 217.844670395618
Depol ratio:         0.556979266301141
'''
import numpy
import jax
from pyscf.data import elements
from pyscfad import gto, scf, cc
from pyscfad.lib import numpy as np

unit2cm = 5140.4871384933695
BOHR = 0.52917721092

mol = gto.Mole()
mol.atom = '''B  ,  0.   0.   0.
              H  ,  0.   0.   2.36328'''
mol.basis = 'aug-cc-pvdz'
mol.unit = 'B'
mol.build(trace_exp=False, trace_ctr_coeff=False)
natm = mol.natm

# CCSD energy
def energy(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = cc.RCCSD(mf)
    mycc.kernel()
    return mycc.e_tot

# hessian in cartesian coordinates
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

# CCSD energy with external electric field applied
def apply_E(mol, E):
    mf = scf.RHF(mol)
    h1 = mf.get_hcore()
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    mf.get_hcore = lambda *args, **kwargs: h1 + np.einsum('x,xij->ij', E, ao_dip)
    mf.kernel()
    mycc = cc.RCCSD(mf)
    mycc.kernel()
    return mycc.e_tot

# zero field
E0 = numpy.zeros((3))
# Raman tensor in au
chi = -jax.jacrev(jax.jacrev(jax.jacrev(apply_E,1),1),0)(mol, E0).coords
# Ramman tensor in Angstrom^2
chi *= BOHR**2

# norm_mode[5] is bond stretching
alpha = numpy.einsum('ijnx,nx->ij', chi, norm_mode[5])
alpha2 = (1./3. * numpy.trace(alpha)) ** 2

alpha_diag = numpy.diag(alpha)
alpha_ij = alpha_diag[:,None] - alpha_diag[None,:]
gamma2  = numpy.einsum('ij->', 1./4. * alpha_ij ** 2)
gamma2 += numpy.einsum('ij->', 1.5 * alpha ** 2)
gamma2 -= numpy.einsum('ij->', 1.5 * numpy.diag(alpha_diag**2))

I = 45 * alpha2 + 7 * gamma2
print(f'Raman activity in A^4/amu: {I}')
rho = 3 * gamma2 / (45 * alpha2 + 4 * gamma2)
print(f'Depolarization ration: {rho}')
