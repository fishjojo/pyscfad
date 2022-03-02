import jax
from pyscfad import gto, scf, cc
'''
Reference nuclear gradient
[[ 3.51278232e-16  4.30055019e-17 -1.15101379e-01]
 [-3.51278232e-16 -4.30055019e-17  1.15101379e-01]]
'''
mol = gto.Mole()
mol.atom = 'H 0. 0. 0.; F 0. 0. 1.1'
mol.basis = 'ccpvdz'
mol.verbose = 5
mol.incore_anyway = True
mol.build()

def ccsd(mol, dm0=None, t1=None, t2=None):
    mf = scf.RHF(mol)
    mf.kernel(dm0)
    mycc = cc.RCCSD(mf)
    mycc.kernel(t1=t1, t2=t2)
    return mycc.e_tot

jac = jax.jacrev(ccsd)(mol)
print(jac.coords)
