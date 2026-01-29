import pytest
import jax
from pyscfad import gto, scf
from pyscfad.mp import dfmp2
from pyscfad.cc import dfccsd
from pyscfad import config
from pyscfad.lno import ccsd as lnoccsd

config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)
config.update('pyscfad_ccsd_implicit_diff', True)

def _test_lnoccsdt_energy_grad(mol, thresh, frozen, lo_type, no_type,
                               tol_e, tol_g, frag_lolist=None):
    def energy0(mol):
        mf = scf.RHF(mol).density_fit()
        mf.kernel()
        mycc = dfccsd.RCCSD(mf, frozen=frozen)
        eris = mycc.ao2mo()
        mycc.kernel(eris=eris)
        et = mycc.ccsd_t(eris=eris)
        return mycc.e_tot + et

    def energy(mol):
        mf = scf.RHF(mol).density_fit()
        ehf = mf.kernel()

        mmp = dfmp2.MP2(mf, frozen=frozen)
        mmp.kernel(with_t2=False)

        mfcc = lnoccsd.LNOCCSD(mf, thresh=thresh, frozen=frozen)
        mfcc.thresh_occ = thresh
        mfcc.thresh_vir = thresh
        mfcc.lo_type = lo_type
        mfcc.no_type = no_type
        mfcc.use_local_virt = True
        mfcc.ccsd_t = True
        mfcc.kernel(frag_lolist=frag_lolist)

        ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)
        return ehf + ecc_pt2corrected

    e, jac = jax.value_and_grad(energy)(mol)
    e0, jac0 = jax.value_and_grad(energy0)(mol)
    assert abs(e-e0) < tol_e
    assert abs(jac.coords - jac0.coords).max() < tol_g

def test_water_dimer():
    atom = '''
     O   -1.485163346097   -0.114724564047    0.000000000000
     H   -1.868415346097    0.762298435953    0.000000000000
     H   -0.533833346097    0.040507435953    0.000000000000
     O    1.416468653903    0.111264435953    0.000000000000
     H    1.746241653903   -0.373945564047   -0.758561000000
     H    1.746241653903   -0.373945564047    0.758561000000
    '''
    basis = '631G'
    mol = gto.Mole(atom=atom, basis=basis)
    mol.verbose = 0
    mol.max_memory = 7000
    mol.incore_anyway = True
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    _test_lnoccsdt_energy_grad(mol, 0, None, 'iao', 'ie',
                               1e-6, 1e-6)

    _test_lnoccsdt_energy_grad(mol, 0, None, 'boys', 'ie',
                               1e-6, 1e-6)

    _test_lnoccsdt_energy_grad(mol, 0, None, 'pm', 'ie',
                               1e-6, 1e-6)

    _test_lnoccsdt_energy_grad(mol, 0, None, 'ibo', 'ie',
                               1e-6, 1e-6)

def test_benzene():
    atom = '''
    C -1.207353289  -0.697065746  0
    C -1.207353289   0.697065746  0
    C  0.000000000   1.394131493  0
    C  1.207353289   0.697065746  0
    C  1.207353289  -0.697065746  0
    C  0.000000000  -1.394131493  0
    H -2.142871219  -1.237187275  0
    H -2.142871219   1.237187275  0
    H  0.000000000   2.474374550  0
    H  2.142871219   1.237187275  0
    H  2.142871219  -1.237187275  0
    H  0.000000000  -2.474374550  0
    '''
    basis = 'sto3g'
    mol = gto.Mole(atom=atom, basis=basis)
    mol.verbose = 0
    mol.max_memory = 7000
    mol.incore_anyway = True
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    # NOTE gradients have small errors for systems with symmetry
    _test_lnoccsdt_energy_grad(mol, 0, None, 'iao', 'ie',
                               1e-6, 1e-6)
    _test_lnoccsdt_energy_grad(mol, 0, None, 'boys', 'ie',
                               1e-6, 1e-6, frag_lolist='1o')
    _test_lnoccsdt_energy_grad(mol, 0, None, 'pm', 'ie',
                               1e-6, 1e-6, frag_lolist='1o')


def test_n2():
    atom = '''
    N 0 0 0
    N 0 0 1.1
    '''
    basis = '631G'
    mol = gto.Mole(atom=atom, basis=basis)
    mol.verbose = 0
    mol.max_memory = 7000
    mol.incore_anyway = True
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    _test_lnoccsdt_energy_grad(mol, 0, None, 'iao', 'ie',
                               1e-6, 1e-6)

    _test_lnoccsdt_energy_grad(mol, 0, None, 'boys', 'ie',
                               1e-6, 1e-6)
