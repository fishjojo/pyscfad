import jax
import pyscf
from pyscfad import gto, scf, mp

"""
Analytic nuclear gradient for MP2 computed by auto-differentiation
"""

mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
mol.basis = '631g'
mol.verbose=5
mol.build(trace_coords=True, trace_exp=True, trace_ctr_coeff=True)

mf = scf.RHF(mol)
mf.kernel()

mymp = mp.MP2(mf)
def func(mymp):
    mymp.reset()
    dm0 = mymp._scf.make_rdm1()
    mymp._scf.kernel(dm0=dm0)
    mymp.mo_coeff = mymp._scf.mo_coeff
    mymp.mo_occ = mymp._scf.mo_occ
    mymp.kernel()
    return mymp.e_tot

jac = jax.jacfwd(func)(mymp)
print(jac._scf.mol.coords)
print(jac._scf.mol.exp)
print(jac._scf.mol.ctr_coeff)
