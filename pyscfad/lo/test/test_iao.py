import jax
from pyscf.data.nist import BOHR
from pyscfad import gto, scf
from pyscfad.lo import iao, orth
from pyscfad import config
#config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)

def _iao(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    orbocc = mf.mo_coeff[:,mf.mo_occ>1e-6]
    c = iao.iao(mol, orbocc)
    c = orth.vec_lowdin(c, mf.get_ovlp())
    return c

def test_iao():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = '631G'
    mol.verbose = 0
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    jac = jax.jacrev(_iao)(mol)
    g0 = jac.coords[:,:,0,2]

    mol.set_geom_('O 0. 0.  0.001; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587')
    c1 = _iao(mol)

    mol.set_geom_('O 0. 0. -0.001; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587')
    c2 = _iao(mol)

    g1 = (c1 - c2) / (0.002 / BOHR)
    assert abs(g0-g1).max() < 1e-6
