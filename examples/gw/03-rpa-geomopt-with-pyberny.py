'''
Geometry optimization using RPA gradients calculated with PySCFAD and PyBerny
as a molecular geometry optimizer
'''
import jax
from pyscf import df as pyscf_df
from pyscfad import gto, dft, df
from pyscfad.gw import rpa
from pyscf.geomopt.berny_solver import optimize, to_berny_geom
from berny import Berny, geomlib
import warnings
warnings.simplefilter("ignore")

def energy_(mol, with_df):
    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel(dm0=None)

    mymp = rpa.RPA(mf)
    mymp.with_df = with_df
    mymp.kernel()
    return mymp.e_tot

def solver(geom, val_and_grad):
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = geom
    mol.basis = 'ccpvdz'
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    auxbasis = pyscf_df.addons.make_auxbasis(mol, mp2fit=True)
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    with_df = df.DF(mol, auxmol=auxmol)

    e_tot, jac = val_and_grad(mol, with_df)

    return e_tot, jac[0].coords + jac[1].mol.coords + jac[1].auxmol.coords


# fictitious molecule to initialize PyBerny optimizer
mol_ = gto.Mole()
mol_.build(atom = 'H 0 0 0; H 0 0 1.', basis = 'sto-3g')
geom = to_berny_geom(mol_)
optimizer = Berny(geom) # initialize geometry optimizer

val_and_grad = jax.value_and_grad(energy_, (0,1))

for iter_, geom in enumerate(optimizer):
    energy, gradients = solver(list(geom), val_and_grad)
    optimizer.send((energy, gradients))
    print(f'iter={iter_+1}   energy={energy:.10f}')

print('\nOptimized geometry:')
print(geom.coords)

'''
iter=1   energy=-1.1691629669
iter=2   energy=-1.1857669916
iter=3   energy=-1.1780975967
iter=4   energy=-1.1915441270
iter=5   energy=-1.1919771635
iter=6   energy=-1.1920713876
iter=7   energy=-1.1920721970

Optimized geometry:
[[0.         0.         0.11749648]
 [0.         0.         0.88250352]]
'''
