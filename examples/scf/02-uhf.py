#!/usr/bin/env python
#
# Author: Junjie Yang
#

import numpy

import pyscf
from   pyscfad import gto, scf

#
# 1. open-shell system
#
mol = gto.Mole()
mol.atom = '''
O 0.0  0     0
H 0.0 -2.757 2.587
H 0.0  2.757 2.587
'''
mol.basis = 'cc-pvdz'
mol.spin   = 1
mol.charge = 1
mol.build()

mf = scf.UHF(mol)
mf.kernel()
jac = mf.energy_grad()
g1  = jac.coords

grad = mf.nuc_grad_method()
g2  = grad.kernel()

assert numpy.linalg.norm(g1-g2) < 1e-6

#
# 2. closed-shell system
#
mol = gto.Mole()
mol.atom = '''
O 0.0  0     0
H 0.0 -2.757 2.587
H 0.0  2.757 2.587
'''
mol.basis = 'cc-pvdz'
mol.spin   = 0
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.kernel()
jac = mf.energy_grad()
g1  = jac.coords

grad = mf.nuc_grad_method()
g2   = grad.kernel()

assert numpy.linalg.norm(g1-g2) < 1e-6

mf = scf.UHF(mol)
mf.kernel()
jac = mf.energy_grad()
g1  = jac.coords

grad = mf.nuc_grad_method()
g2  = grad.kernel()

assert numpy.linalg.norm(g1-g2) < 1e-6
