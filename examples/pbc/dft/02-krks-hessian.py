import jax
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
cell.build(trace_exp=False, trace_ctr_coeff=False)

def ehf(cell):
    kpts = cell.make_kpts([2,1,1])
    mf = dft.KRKS(cell, kpts=kpts, exxdiv=None)
    mf.xc = 'pbe'
    e = mf.kernel()
    return e
jac = jax.hessian(ehf)(cell)
print(jac.coords.coords)
