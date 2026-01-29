'''LNO-CCSD(T) energy and gradient
'''
import jax
from pyscfad import gto, scf, mp, cc
from pyscfad.cc import dfccsd
from pyscfad import config
from pyscfad.lno import LNOCCSD

config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)
config.update('pyscfad_ccsd_implicit_diff', True)

atom = 'water_dimer.xyz'
basis = 'ccpvdz'
frozen = 2

mol = gto.Mole(atom=atom, basis=basis)
mol.verbose = 4
mol.build(trace_exp=False, trace_ctr_coeff=False)

# canonical CCSD(T)
mf = scf.RHF(mol).density_fit()
mf.kernel()
mycc = dfccsd.RCCSD(mf, frozen=frozen)
eris = mycc.ao2mo()
mycc.kernel(eris=eris)
et = mycc.ccsd_t(eris=eris)

# LNO-CCSD(T)
thresh = 1e-4
def energy(mol):
    mf = scf.RHF(mol).density_fit()
    ehf = mf.kernel()

    mmp = mp.dfmp2.MP2(mf, frozen=frozen)
    mmp.kernel(with_t2=False)

    mfcc = LNOCCSD(mf, thresh=thresh, frozen=frozen)
    mfcc.thresh_occ = thresh
    mfcc.thresh_vir = thresh
    mfcc.lo_type = 'iao'
    mfcc.no_type = 'ie'
    mfcc.ccsd_t = True
    mfcc.kernel(frag_lolist=None)

    ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)
    return ehf + ecc_pt2corrected

e, jac = jax.value_and_grad(energy)(mol)
print(e, mycc.e_tot+et, e-(mycc.e_tot+et))
print(jac.coords)
