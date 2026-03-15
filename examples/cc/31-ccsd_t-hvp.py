"""
Nuclear Hessian gradient product of CCSD(T)
"""
import jax
from pyscfad import gto, scf
from pyscfad.cc import rccsd
from pyscfad import config

config.update("pyscfad_scf_implicit_diff", True)
config.update("pyscfad_ccsd_implicit_diff", True)

mol = gto.Mole()
mol.atom = """
    O  0.000000  0.000000   0.000000
    H  0.758602  0.000000   0.504284
    H  0.758602  0.000000  -0.504284
"""
mol.basis = "ccpvtz"
mol.verbose = 4
mol.incore_anyway = True
mol.max_memory = 24000
mol.build(trace_exp=False, trace_ctr_coeff=False)

def energy(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = rccsd.RCCSD(mf)
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    et = mycc.ccsd_t(eris=eris)
    return mycc.e_tot + et

# function for computing the Jacobian
jac = lambda x, *args: jax.jacrev(energy)(x)
# function for computing the Hessian vector product
hessp = lambda x, p, *args: jax.vjp(jac, x)[1](p)[0]

g = jac(mol)
print("Nuclear gradient:")
print(g.coords)

# prepare a Mole object which has its coords
# attribute set as the nuclear gradient
mol1 = mol.copy()
mol1.coords = g.coords
hvp = hessp(mol, mol1)
print("Nuclear Hessian gradient product:")
print(hvp.coords)
