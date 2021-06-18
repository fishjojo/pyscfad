import jax.scipy as scipy
from pyscfad.lib import numpy as jnp
from pyscfad.lib.numpy_helper import unpack_triu

def update_rotate_matrix(dx, u0=1):
    dr = unpack_triu(dx, filltril=2)
    u = jnp.dot(u0, scipy.linalg.expm(dr))
    return u

def rotate_mo(mo_coeff, u):
    mo = jnp.dot(mo_coeff, u)
    return mo

def rotate_mo1(mo_coeff, x):
    u = update_rotate_matrix(x)
    mo_coeff1 = rotate_mo(mo_coeff, u)
    return mo_coeff1
