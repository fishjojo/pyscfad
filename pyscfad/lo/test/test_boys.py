import jax
from pyscf.data.nist import BOHR
from pyscfad import gto, scf
from pyscfad.lo import boys
#from pyscfad import config
#config.update('pyscfad_moleintor_opt', True)
#config.update('pyscfad_scf_implicit_diff', True)

def _boys(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    orbocc = mf.mo_coeff[:,mf.mo_occ>1e-6]
    c = boys.boys(mol, orbocc, init_guess='atomic')
    return c

#FIXME not stable
def test_boys_skip():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = '631G'
    mol.verbose = 0
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    jac = jax.jacrev(_boys)(mol)
    g0 = jac.coords[:,:,0,2]

    mol.set_geom_('O 0. 0.  0.001; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587')
    c1 = _boys(mol)

    mol.set_geom_('O 0. 0. -0.001; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587')
    c2 = _boys(mol)

    g1 = (c1 - c2) / (0.002 / BOHR)
    assert abs(g0-g1).max() < 2e-4
