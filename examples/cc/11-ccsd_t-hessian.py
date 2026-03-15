"""
Nuclear Hessian of CCSD(T)
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
mol.max_memory = 80000
mol.build(trace_exp=False, trace_ctr_coeff=False)

def energy(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = rccsd.RCCSD(mf)
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    et = mycc.ccsd_t(eris=eris)
    return mycc.e_tot + et

hess = jax.jacrev(jax.grad(energy))(mol)
print("Nuclear Hessian:")
print(hess.coords.coords)
