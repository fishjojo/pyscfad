"""
Nuclear gradient of density-fitting CCSD(T)

Reference:
CCSD(T) total energy: -76.30586034
Nuclear gradient:
[[ 5.81461842e-02  6.30418580e-17  8.76964385e-15]
 [-2.90730921e-02 -3.65174463e-17 -1.27052571e-01]
 [-2.90730921e-02 -2.65244117e-17  1.27052571e-01]]
"""
import jax
from pyscfad import gto, scf
from pyscfad.cc import dfccsd
from pyscfad import config

# Setting `pyscfad_moleintor_opt` to `True` will use the
# efficient back-propagation CPU implementation. However, only
# 1st order derivative is available.
config.update("pyscfad_moleintor_opt", True)
config.update("pyscfad_scf_implicit_diff", True)
config.update("pyscfad_ccsd_implicit_diff", True)

mol = gto.Mole()
mol.atom = '''
    O  0.000000  0.000000  0.000000
    H  0.758602  0.000000  0.504284
    H  0.758602  0.000000  -0.504284
'''
mol.basis = 'aug-ccpvtz'
mol.verbose = 4
mol.incore_anyway = True
mol.max_memory = 16000
mol.build(trace_exp=False, trace_ctr_coeff=False)

def energy(mol):
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    mycc = dfccsd.RCCSD(mf)
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    et = mycc.ccsd_t(eris=eris)
    return mycc.e_tot + et

e, jac = jax.value_and_grad(energy)(mol)
print(f"CCSD(T) total energy: {e:.8f}")
print("Nuclear gradient:")
print(jac.coords)
