'''
Orbital optimized RPA
'''
import numpy
from scipy.optimize import minimize
import jax
from jax import numpy as np
from pyscf import df as pyscf_df
from pyscfad import gto, dft, df
from pyscfad.gw import rpa
from pyscfad.tools import rotate_mo1

mol = gto.Mole()
mol.verbose = 3
mol.atom = [
    [2 , (0. , 0.     , 0.)],
    [2 , (0. , 0. , 2.6)]]
mol.basis = 'def2-svp'
mol.build(trace_exp=False, trace_ctr_coeff=False)

mf = dft.RKS(mol)
mf.xc = 'pbe'
mf.kernel()

mymp=rpa.RPA(mf)
mymp.kernel()
print("PBE-RPA energy:", mymp.e_tot)

auxbasis = pyscf_df.addons.make_auxbasis(mol, mp2fit=True)
auxmol = df.addons.make_auxmol(mol, auxbasis)

nocc = 1
def energy(mol, auxmol, x):
    x = np.asarray(x)
    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel(dm0=None)

    mo_coeff = rotate_mo1(mf.mo_coeff, x)
    mf.mo_coeff = mo_coeff

    mymp = rpa.RPA(mf)
    mymp.mo_coeff = mo_coeff
    mymp.with_df = df.DF(mol, auxmol=auxmol)
    mymp.kernel()
    return mymp.e_tot

nao = mol.nao
size = nao*(nao+1)//2
x0 = numpy.zeros([size,])

def func(x0, mol, auxmol):
    def grad(x0, mol, auxmol):
        f, g = jax.value_and_grad(energy, 2)(mol, auxmol, x0)
        return f, g

    f, g = grad(x0, mol, auxmol)
    print("energy:", f, "norm g:", numpy.linalg.norm(g))
    return (numpy.asarray(f), numpy.asarray(g))

options ={"disp": True, "gtol": 1e-5}
res = minimize(func, x0, args=(mol, auxmol), jac=True, method="BFGS", options = options)
e,g = func(res.x, mol, auxmol)
print("PBE-OO-RPA energy:", e)
