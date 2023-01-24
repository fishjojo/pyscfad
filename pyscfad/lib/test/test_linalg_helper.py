import pytest
import jax
from jax import scipy as jscipy
from pyscf import numpy as np
from pyscfad.lib import linalg_helper as linalg

def test_eigh():
    a = np.ones((2,2))
    b = np.eye(2)

    w0, v0 = jscipy.linalg.eigh(a) 
    w, v = linalg.eigh(a)

    assert abs(w-w0).max() < 1e-10
    assert abs(v-v0).max() < 1e-10

    jac0 = jax.jacfwd(jscipy.linalg.eigh)(a)
    jac = jax.jacfwd(linalg.eigh)(a)

    assert abs(jac[0] - jac0[0]).max() < 1e-10
    assert abs(jac[1] - jac0[1]).max() < 1e-10

    #finite difference
    m = -0.0005
    p = 0.0005
    b_m = np.array([[1., 0.],[0., 1.+m]])
    w_m, v_m = linalg.eigh(a, b_m)
    b_p = np.array([[1, 0.],[0., 1.+p]])
    w_p, v_p = linalg.eigh(a, b_p)

    g0 = (-v_p - v_m) / 0.001 # -v_p due to specific gauge

    jac = jax.jacfwd(linalg.eigh, argnums=1)(a, b)
    g = jac[1][:,:,1,1]
    assert abs(g-g0).max() < 1e-7
