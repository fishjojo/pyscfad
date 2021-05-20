import jax
from pyscfad.pbc import gto
from pyscfad.lib import numpy as jnp

cell = gto.Cell()
cell.atom = '''Si 0.,  0.,  0.
               Si 1.3467560987,  1.3467560987,  1.3467560987'''
cell.a = '''0.            2.6935121974    2.6935121974
            2.6935121974  0.              2.6935121974
            2.6935121974  2.6935121974    0.    '''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.build(trace_coords=True)
kpts = cell.make_kpts([2,1,1])

def func(cell):
    res = 0.
    s1 = cell.pbc_intor("int1e_ovlp", kpts=kpts)
    for s in s1:
        res += jnp.sum(s*s.conj())
    return res.real

_, func_vjp = jax.vjp(func, cell)
jac = func_vjp(1.)
print(jac[0].coords)

jac = jax.jacfwd(func)(cell)
print(jac.coords)

jac = jax.grad(func)(cell)
print(jac.coords)
