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
cell.verbose = 5
cell.build(trace_coords=True)

mf = scf.RHF(cell)
#mf.kernel()
jac = mf.mol_grad_ad(mode='fwd')
print(jac.coords)
