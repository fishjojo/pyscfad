import jax
from pyscfad import gto, scf, cc
'''
Reference nuclear gradient
[[0.0, 0.0, -1.15101379e-01]
 [0.0, 0.0,  1.15101379e-01]]

Note that without implicit differentiation turned on,
the gradient will be initial guess dependent.
'''
mol = gto.Mole()
mol.atom = 'H 0. 0. 0.; F 0. 0. 1.1'
mol.basis = 'ccpvdz'
mol.verbose = 3
mol.incore_anyway = True
mol.build()

def ccsd(mol, dm0=None, t1=None, t2=None):
    mf = scf.RHF(mol)
    mf.kernel(dm0)
    mycc = cc.RCCSD(mf)
    mycc.kernel(t1=t1, t2=t2)
    return mycc.e_tot

jac = jax.jacrev(ccsd)(mol)
print(f'Nuclaer gradient:\n{jac.coords}')
print(f'Gradient wrt basis exponents:\n{jac.exp}')
print(f'Gradient wrt basis contraction coefficients:\n{jac.ctr_coeff}')

mf = scf.RHF(mol)
mf.kernel()
mycc = cc.RCCSD(mf)
mycc.kernel()
jac = jax.jacrev(ccsd)(mol, t1=mycc.t1, t2=mycc.t2)
print(f'Nuclaer gradient:\n{jac.coords}')
print(f'Gradient wrt basis exponents:\n{jac.exp}')
print(f'Gradient wrt basis contraction coefficients:\n{jac.ctr_coeff}')
