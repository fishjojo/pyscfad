'''
Orbital optimized density-fitted RPA
'''
import numpy
from scipy.optimize import minimize
import jax
from pyscf.df.addons import make_auxbasis
from pyscfad import gto, dft, df
from pyscfad.gw import rpa
from pyscfad.tools import rotate_mo1

# molecular structure
mol = gto.Mole()
mol.atom = [['He', (0., 0., 0.)],
            ['He', (0., 0., 2.6)]]
mol.basis = 'def2-svp'
mol.build()

# RKS/PBE
mf = dft.RKS(mol, xc='PBE')
mf.kernel()
mo_coeff = mf.mo_coeff

# density fitting basis
dfobj = df.DF(mol, make_auxbasis(mol, mp2fit=True))

def rpa_energy(x):
    # apply orbital rotation
    mf.mo_coeff = rotate_mo1(mo_coeff, x)
    # density-fitted RPA
    myrpa = rpa.RPA(mf)
    myrpa.with_df = dfobj
    myrpa.kernel()
    return myrpa.e_tot

x0 = numpy.zeros([mol.nao*(mol.nao+1)//2,])
value_and_grad = lambda x : jax.value_and_grad(rpa_energy)(x)
res = minimize(value_and_grad, x0, jac=True,
               method="BFGS", options={'gtol': 1e-5})
etot = rpa_energy(res.x)
print("OO-RPA/PBE energy:", etot)
