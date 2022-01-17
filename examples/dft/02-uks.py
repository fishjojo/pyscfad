import numpy

import pyscf
from pyscf   import scf
from pyscfad import gto, dft

def init_guess_mixed(mf, mixing_parameter=numpy.pi/4):
    mol  = mf.mol
    h1e  = scf.hf.get_hcore(mol)
    s1e  = scf.hf.get_ovlp(mol)

    dm0  = mf.get_init_guess(key='minao')
    vhf  = mf.get_veff(dm=dm0)
    fock = mf.get_fock(h1e, s1e, vhf, dm0, 0, None)

    mo_energy, mo_coeff = mf.eig(fock, s1e)
    nao, nmo = mo_coeff[0].shape

    mo_occ   = mf.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)

    homo_idx_a = mol.nelec[0] - 1
    lumo_idx_a = homo_idx_a + 1

    homo_idx_b = mol.nelec[1] - 1
    lumo_idx_b = homo_idx_b + 1

    assert mo_occ[0][homo_idx_a] == 1.0 
    assert mo_occ[0][lumo_idx_a] == 0.0
    assert mo_occ[1][homo_idx_b] == 1.0 
    assert mo_occ[1][lumo_idx_b] == 0.0

    theta   = mixing_parameter
    r_theta = numpy.array(
      [[numpy.cos(theta), -numpy.sin(theta)],
       [numpy.sin(theta),  numpy.cos(theta)]]
       )
    t_theta = [numpy.eye(nmo, nmo), numpy.eye(nmo, nmo)]

    t_theta[0][homo_idx_a:lumo_idx_a+1, homo_idx_a:lumo_idx_a+1] = r_theta
    t_theta[1][homo_idx_b:lumo_idx_b+1, homo_idx_b:lumo_idx_b+1] = r_theta.T
    
    mo_coeff = numpy.einsum("spq,sqn->spn", t_theta, mo_coeff)
    return scf.uhf.make_rdm1(mo_coeff, mo_occ)


mol      = gto.Mole()
mol.atom = '''
  H    0.0000000    0.0000000    1.00000
  H    0.0000000    0.0000000   -1.00000
'''
mol.basis   = 'cc-pvdz'
mol.verbose = 3
mol.build()

# LDA
mf    = dft.UKS(mol)
mf.xc = "LDA"
dm0   = init_guess_mixed(mf, mixing_parameter=numpy.pi/4)
mf.kernel(dm0=dm0)

jac = mf.energy_grad()
g1  = jac.coords

grad = mf.nuc_grad_method()
grad.verbose = 0
g2   = grad.kernel()

assert abs(g1-g2).max() < 1e-6

# GGA
mf    = dft.UKS(mol)
mf.xc = "PBE"
mf.kernel(dm0=dm0)

jac = mf.energy_grad()
g1  = jac.coords

grad = mf.nuc_grad_method()
grad.verbose = 0
g2   = grad.kernel()

assert abs(g1-g2).max() < 1e-6

# meta-GGA
# mf    = dft.UKS(mol)
# mf.xc = "TPSS"
# mf.kernel(dm0=dm0)

# jac = mf.energy_grad()
# g1  = jac.coords

# g2   = numpy.asarray(
#         [[0.0, 0.0,  2.64877520e-02],
#          [0.0, 0.0, -2.64877520e-02]]
# ) # From finite-difference

# assert abs(g1-g2).max() < 1e-6