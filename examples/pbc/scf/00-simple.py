import jax
from pyscfad.pbc import gto
from pyscfad.pbc import scf, df

cell = gto.Cell()
cell.atom = '''Si 0.,  0.,  0.
               Si 1.3467560987,  1.3467560987,  1.3467560987'''
cell.a = '''0.            2.6935121974    2.6935121974
            2.6935121974  0.              2.6935121974
            2.6935121974  2.6935121974    0.    '''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.build(trace_coords=True)

mf = scf.RHF(cell)
jac = jax.jacrev(mf.__class__.get_hcore)(mf)
print(jac.cell.coords)
