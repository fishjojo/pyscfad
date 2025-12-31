"""Gaussian integral derivatives
"""
import jax
from pyscfad import gto

mol = gto.Mole()
mol.atom = 'H 0. 0. 0.0; Li 0. 0. 0.74'
mol.basis = 'sto3g'
mol.build()

int_fn = lambda mol, intor: mol.intor(intor)

jac = jax.jacrev(int_fn)(mol, "int1e_nuc")
# jacobian w.r.t. atom centers
print(jac.coords.shape)
# jacobian w.r.t. basis exponents
print(jac.exp.shape)
# jacobian w.r.t. basis contraction coefficients
print(jac.ctr_coeff.shape)

hess = jax.hessian(int_fn)(mol, "int1e_nuc")
# hessian w.r.t. atom centers
print(hess.coords.coords.shape)
