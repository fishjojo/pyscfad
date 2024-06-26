'''
stress tensor
----------------------------
[[-7.57812381e-01  1.65085403e-09  1.65085540e-09]
 [ 1.65085006e-09 -7.57812381e-01  1.65084496e-09]
 [ 1.65084745e-09  1.65084512e-09 -7.57812381e-01]]
[[-3.59347869e-03  7.82820248e-12  7.82820901e-12]
 [ 7.82818367e-12 -3.59347869e-03  7.82815947e-12]
 [ 7.82817132e-12  7.82816026e-12 -3.59347869e-03]]
[[-6.59876007e-01  1.43750484e-09  1.43750604e-09]
 [ 1.43750138e-09 -6.59876007e-01  1.43749694e-09]
 [ 1.43749911e-09  1.43749708e-09 -6.59876007e-01]]
----------------------------
'''
import numpy
import jax
from jax import numpy as np
from pyscf.data.nist import BOHR, HARTREE2EV
from pyscfad.pbc import gto as pbcgto
from pyscfad.pbc import scf as pbcscf

#aas = numpy.arange(5,6.0,0.1,dtype=float)
for aa in [5.]:
    basis = 'gth-szv'
    pseudo = 'gth-pade'
    lattice = numpy.asarray([[0., aa/2, aa/2],
              [aa/2, 0., aa/2],
              [aa/2, aa/2, 0.]])
    atom = [['Si', [0., 0., 0.]],
            ['Si', [aa/4, aa/4, aa/4]]]

    strain = np.zeros((3,3))
    def khf_energy(strain, lattice):
        cell = pbcgto.Cell()
        cell.atom = atom
        cell.a = lattice
        cell.basis = basis
        cell.pseudo = pseudo
        cell.verbose = 4
        cell.exp_to_discard=0.1
        cell.mesh = [21]*3
        cell.max_memory=100000
        cell.build(trace_lattice_vectors=True)

        cell.abc += np.einsum('ab,nb->na', strain, cell.lattice_vectors())
        cell.coords += np.einsum('xy,ny->nx', strain, cell.atom_coords())

        kpts = cell.make_kpts([2,2,2])

        mf = pbcscf.KRHF(cell, kpts=kpts, exxdiv=None)
        ehf = mf.kernel()
        return ehf, cell

    jac, cell = jax.jacrev(khf_energy, has_aux=True)(strain, lattice)
    print('stress tensor')
    print('----------------------------')
    print(jac)
    print(jac / cell.vol)
    print(jac*HARTREE2EV / (cell.vol*(BOHR**3)))
    print('----------------------------')
