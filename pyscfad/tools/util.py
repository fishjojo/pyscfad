from jax import scipy
from pyscf import numpy as np
from pyscfad.lib.numpy_helper import unpack_triu

def update_rotate_matrix(dx, u0=1):
    dr = unpack_triu(dx, filltril=2)
    u = np.dot(u0, scipy.linalg.expm(dr))
    return u

def rotate_mo(mo_coeff, u):
    mo = np.dot(mo_coeff, u)
    return mo

def rotate_mo1(mo_coeff, x):
    u = update_rotate_matrix(x)
    mo_coeff1 = rotate_mo(mo_coeff, u)
    return mo_coeff1

def rotate_mo1_ov(mo_coeff, x, nocc):
    u = update_rotate_matrix(x)
    u = u.at[:nocc,:nocc].set(0.)
    u = u.at[nocc:,nocc:].set(0.)
    mo_coeff1 = rotate_mo(mo_coeff, u)
    return mo_coeff1
