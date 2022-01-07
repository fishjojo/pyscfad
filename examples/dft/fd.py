import numpy

import pyscf
from pyscf import gto, dft

"""
Analytic nuclear gradient for RKS computed by auto-differentiation
"""

for dd in [1e-3, 1e-4, 1e-5]:
    r = 0.3540000 - dd
    mol      = gto.Mole()
    mol.atom = f'''
    H    0.0000000    0.0000000    { r:16.8f}
    H    0.0000000    0.0000000    {-r:16.8f}
    '''
    mol.basis   = 'cc-pvdz'
    mol.verbose = 0
    mol.build()

    # meta-GGA
    mf    = dft.RKS(mol)
    mf.xc = "TPSS"
    mf.kernel()

    coords1 = mol.atom_coords()
    e1      = mf.e_tot

    r = 0.3540000 + dd
    mol      = gto.Mole()
    mol.atom = f'''
    H    0.0000000    0.0000000    { r:16.8f}
    H    0.0000000    0.0000000    {-r:16.8f}
    '''
    mol.basis   = 'cc-pvdz'
    mol.verbose = 0
    mol.build()

    # meta-GGA
    mf    = dft.RKS(mol)
    mf.xc = "TPSS"
    mf.kernel()

    coords2 = mol.atom_coords()
    e2      = mf.e_tot

    dx = coords2 - coords1
    dx[numpy.abs(dx) < 1e-8] = 1e8

    print((e2-e1)/dx/2)