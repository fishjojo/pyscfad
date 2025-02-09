import jax
from pyscfad import gto,scf,cc

'''
Analytic nuclear gradients:
[[ 3.31592012e-18  2.69897541e-18 -8.60709468e-02]
 [-3.31592012e-18 -2.69897541e-18  8.60709468e-02]]
'''

mol = gto.Mole()
mol.atom = 'H 0. 0. 0.; F 0. 0. 1.1'
mol.basis = '631g'
mol.verbose = 5
mol.incore_anyway = True
mol.build(trace_exp=False, trace_ctr_coeff=False)

def energy(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = cc.RCCSD(mf)
    mycc.kernel()
    et = mycc.ccsd_t()
    return mycc.e_tot + et

g1 = jax.jacrev(energy)(mol).coords
print(g1)
