import numpy

import pyscf
from pyscf   import scf
from pyscf   import gto, dft

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

for dd in [1e-3, 1e-4, 1e-5]:
    r = 1.00000 - dd
    mol      = gto.Mole()
    mol.atom = f'''
    H    0.0000000    0.0000000    { r:16.8f}
    H    0.0000000    0.0000000    {-r:16.8f}
    '''
    mol.basis   = 'cc-pvdz'
    mol.verbose = 3
    mol.build()

    # meta-GGA
    mf    = dft.UKS(mol)
    mf.xc = "TPSS"
    dm0   = init_guess_mixed(mf, mixing_parameter=numpy.pi/4)
    mf.kernel(dm0=dm0)

    coords1 = mol.atom_coords()
    e1      = mf.e_tot

    r = 1.00000 + dd
    mol      = gto.Mole()
    mol.atom = f'''
    H    0.0000000    0.0000000    { r:16.8f}
    H    0.0000000    0.0000000    {-r:16.8f}
    '''
    mol.basis   = 'cc-pvdz'
    mol.verbose = 3
    mol.build()

    # meta-GGA
    mf    = dft.UKS(mol)
    mf.xc = "TPSS"
    dm0   = init_guess_mixed(mf, mixing_parameter=numpy.pi/4)
    mf.kernel(dm0=dm0)

    coords2 = mol.atom_coords()
    e2      = mf.e_tot

    dx = coords2 - coords1
    dx[numpy.abs(dx) < 1e-8] = 1e8

    print((e2-e1)/dx/2)