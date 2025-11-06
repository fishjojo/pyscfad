"""
Interface DFT with autoxc for functional training
"""

import jax
from pyscfad import numpy as np
from pyscfad import gto, dft
from autoxc.api import functionals as fns

mol = gto.Mole()
mol.atom = 'H 0 0 0; Li 0 0 1.6'
mol.basis = '631g*'
mol.verbose=4
mol.build(trace_exp=False, trace_ctr_coeff=False)

# PBE functional
xc = {
    "GGA_X_PBE": {"coeff": 1.0, "params": np.asarray(fns.gga_x_pbe.params)},
    "GGA_C_PBE": {"coeff": 1.0, "params": np.asarray(fns.gga_c_pbe.params)},
}

def energy(mol, xc):
    mf = dft.RKS(mol, xc=xc)
    e = mf.kernel()
    return e

e, g = jax.value_and_grad(energy, 1)(mol, xc)
print(g) # energy gradient w.r.t. functional parameters

g = jax.jacrev(jax.grad(energy, 0), 1)(mol, xc)
print(g.coords) # force gradient w.r.t. functional parameters
