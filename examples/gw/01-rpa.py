'''
RPA e_tot, e_hf, e_corr =  -76.26428191794197 -75.95645187758402 -0.30783004035795963

g_e_tot
finite diff
[[-7.52006046e-12 -7.52006046e-11  2.04900196e-02]
 [-3.00802419e-11  7.11947436e-03 -1.02450070e-02]
 [ 3.00802419e-11 -7.11947447e-03 -1.02450070e-02]]

unroll loop
[[-2.11715820e-17 -5.21369404e-17  2.04900964e-02]
 [ 1.66819790e-17  7.11935407e-03 -1.02450516e-02]
 [ 8.87979279e-18 -7.11935407e-03 -1.02450516e-02]]

implicit function
[[-1.57448496e-16 -1.56984362e-15  2.04890013e-02]
 [ 2.21260007e-17  7.11883505e-03 -1.02445040e-02]
 [ 1.34912502e-16 -7.11883505e-03 -1.02445040e-02]]


g_e_hf
[[-7.52006046e-12 -6.01604837e-11 -2.21207015e-02]
 [-3.00802419e-11 -1.30074871e-02  1.10603549e-02]
 [ 3.00802419e-11  1.30074870e-02  1.10603549e-02]]

g_e_corr
[[ 8.81257086e-14 -1.84476483e-11  4.26107212e-02]
 [ 1.05750850e-12  2.01269614e-02 -2.13053619e-02]
 [-4.99379015e-13 -2.01269615e-02 -2.13053619e-02]]

'''

import jax
from pyscf import df as pyscf_df
from pyscfad import gto, dft, scf, df
from pyscfad.gw import rpa
from pyscfad import config

config.update('pyscfad_scf_implicit_diff', True)
# Using optimized C implementation for gradients calculations.
# This requires the `pyscfadlib` package, which can be installed with
# `pip install pyscfadlib`
#config.update('pyscfad_moleintor_opt', True)

mol = gto.Mole()
mol.verbose = 4
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.7571 , 0.5861)],
    [1 , (0. , 0.7571 , 0.5861)]]
mol.basis = 'def2-svp'
mol.max_memory = 4000
mol.build(trace_exp=False, trace_ctr_coeff=False)

auxbasis = pyscf_df.addons.make_auxbasis(mol, mp2fit=True)
auxmol = df.addons.make_auxmol(mol, auxbasis)
with_df = df.DF(mol, auxmol=auxmol)

def energy(mol, with_df):
    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel(dm0=None)

    mymp = rpa.RPA(mf)
    mymp.with_df = with_df
    mymp.kernel()
    return mymp.e_tot

#print(energy(mol, with_df))

jac = jax.grad(energy, (0,1))(mol, with_df)
print(jac[0].coords + jac[1].mol.coords + jac[1].auxmol.coords) 
