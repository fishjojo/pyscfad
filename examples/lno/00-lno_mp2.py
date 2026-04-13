'''LNO-MP2 energy and gradient.
'''
import jax
from pyscfad import gto, scf
from pyscfad.mp import dfmp2
from pyscfad import config
from pyscfad.lno import LNOMP2

# use optimized backpropagation
config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)
config.update('pyscfad_ccsd_implicit_diff', True)

atom = 'water_dimer.xyz'
basis = 'ccpvdz'
frozen = None

mol = gto.Mole(atom=atom, basis=basis)
mol.verbose = 4
mol.build(trace_exp=False, trace_ctr_coeff=False)

# canonical MP2
def mp2_energy(mol):
    mf = scf.RHF(mol).density_fit()
    mf.kernel()

    mymp = dfmp2.MP2(mf, frozen=frozen)
    mymp.kernel()
    return mymp.e_tot

e_mp2, jac_mp2 = jax.value_and_grad(mp2_energy)(mol)

# LNO-MP2
thresh = 1e-4
def lno_mp2_energy(mol):
    mf = scf.RHF(mol).density_fit()
    ehf = mf.kernel()

    mfcc = LNOMP2(mf, thresh=thresh, frozen=frozen)
    mfcc.thresh_occ = thresh
    mfcc.thresh_vir = thresh
    mfcc.lo_type = 'iao'
    mfcc.kernel(frag_lolist=None)
    return ehf + mfcc.e_corr

e_lno_mp2, jac_lno_mp2 = jax.value_and_grad(lno_mp2_energy)(mol)

print(e_mp2, e_lno_mp2, e_mp2-e_lno_mp2)
print(jac_lno_mp2.coords)
print(abs(jac_lno_mp2.coords - jac_mp2.coords).max())
