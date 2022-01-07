import numpy

import pyscf
from pyscfad import gto, dft

"""
Analytic nuclear gradient for RKS computed by auto-differentiation
"""

mol      = gto.Mole()
mol.atom = '''
  H    0.0000000    0.0000000    0.3540000
  H    0.0000000    0.0000000   -0.3540000
'''
mol.basis   = 'cc-pvdz'
mol.verbose = 3
mol.build()

# LDA
mf    = dft.RKS(mol)
mf.xc = "LDA"
mf.kernel()

jac = mf.energy_grad()
g1  = jac.coords

grad = mf.nuc_grad_method()
grad.verbose = 0
g2   = grad.kernel()

assert abs(g1-g2).max() < 1e-6

# GGA
mf    = dft.RKS(mol)
mf.xc = "PBE"
mf.kernel()

jac = mf.energy_grad()
g1  = jac.coords

grad = mf.nuc_grad_method()
grad.verbose = 0
g2   = grad.kernel()

assert abs(g1-g2).max() < 1e-6

# meta-GGA
mf    = dft.RKS(mol)
mf.xc = "TPSS"
mf.kernel()

jac = mf.energy_grad()
g1  = jac.coords

# MGGA Gradient Not Implemented 
# grad = mf.nuc_grad_method()
# grad.verbose = 0
g2   = numpy.asarray(
        [[0.0, 0.0, -4.14229185e-02],
         [0.0, 0.0,  4.14229185e-02]]
) # From finite-difference

assert abs(g1-g2).max() < 1e-6