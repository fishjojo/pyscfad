import numpy
import jax
from jax import numpy as np
from pyscf.data.nist import BOHR, HARTREE2EV
from pyscfad.pbc import gto as pbcgto
from pyscfad.pbc import scf as pbcscf

aas = numpy.arange(5,6.0,0.1,dtype=float)
for aa in [5.,]:
    basis = 'gth-szv'
    pseudo = 'gth-pade'
    lattice = numpy.asarray([[0., aa/2, aa/2],
              [aa/2, 0., aa/2],
              [aa/2, aa/2, 0.]])
    atom = [['Si', [0., 0., 0.]],
            ['Si', [aa/4, aa/4, aa/4]]]

    cell0 = pbcgto.Cell()
    cell0.atom = atom
    cell0.a = lattice
    cell0.basis = basis
    cell0.pseudo = pseudo
    cell0.verbose = 4
    cell0.exp_to_discard=0.1
    cell0.build()

    coords = []
    for i, a in enumerate(atom):
        coords.append(a[1])
    coords = numpy.asarray(coords)

    strain = numpy.zeros((3,3))
    def khf_energy(strain, lattice, coords):
        cell = pbcgto.Cell()
        cell.atom = atom
        cell.a = lattice
        cell.basis = basis
        cell.pseudo = pseudo
        cell.verbose = 4
        cell.exp_to_discard=0.1
        cell.max_memory=24000
        cell.build(trace_lattice_vectors=True)

        cell.abc += np.einsum('ab,nb->na', strain, cell.lattice_vectors())
        cell.coords += np.einsum('xy,ny->nx', strain, cell.atom_coords())

        kpts = cell.make_kpts([1,1,1])

        mf = pbcscf.KRHF(cell, kpts=kpts, exxdiv=None)
        ehf = mf.kernel(dm0=None)
        return ehf

    jac = jax.grad(khf_energy)(strain, lattice, coords)
    print('stress tensor')
    print('----------------------------')
    print(jac)
    print(jac / cell0.vol)
    print(jac*HARTREE2EV / (cell0.vol*(BOHR**3)))
    print('----------------------------')
