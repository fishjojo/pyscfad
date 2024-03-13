from pyscfad import numpy as np
from pyscfad.soscf.ciah import extract_rotation

def rotate_mo1(mo_coeff, x):
    u = extract_rotation(x)
    return np.dot(mo_coeff, u)

def rotate_mo1_ov(mo_coeff, x, nocc):
    u = extract_rotation(x)
    u = u.at[:nocc,:nocc].set(0)
    u = u.at[nocc:,nocc:].set(0)
    return np.dot(mo_coeff, u)
