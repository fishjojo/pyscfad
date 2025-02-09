'''
--------------- KRHF gradients ---------------
         x                y                z
0 Si    -0.0109162228    -0.0028233781    -0.0028233781
1 Si     0.0109162228     0.0028233781     0.0028233781
----------------------------------------------
'''
import jax
from pyscfad.pbc import gto
from pyscfad.pbc import scf, df

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
cell.verbose = 4
cell.build()
kpts = cell.make_kpts([2,1,1])

def hf_energy(cell, kpts):
    mf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    e_tot = mf.kernel()
    return e_tot
e_tot, jac = jax.value_and_grad(hf_energy)(cell, kpts)
print(f'Nuclaer gradient:\n{jac.coords}')
