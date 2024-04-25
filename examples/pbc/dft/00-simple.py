'''
--------------- KRKS gradients ---------------
         x                y                z
0 Si    -0.0071742877    -0.0071742877    -0.0071742877
1 Si     0.0071739026     0.0071739026     0.0071739026
----------------------------------------------
'''

from pyscfad.pbc import gto
from pyscfad.pbc import dft, df

basis = 'gth-szv'
pseudo = 'gth-pade'

a = 5.431020511
lattice = [[0., a/2, a/2],
          [a/2, 0., a/2],
          [a/2, a/2, 0.]]
disp = 0.01
atom = [['Si', [0., 0., 0.]],
        ['Si', [a/4+disp, a/4+disp, a/4+disp]]]

cell = gto.Cell()
cell.atom = atom
cell.a = lattice
cell.basis = basis
cell.pseudo = pseudo
cell.build()

mf = dft.RKS(cell, exxdiv=None)
mf.xc = 'pbe'
mf.kernel()
jac = mf.energy_grad()
print(jac.coords)
