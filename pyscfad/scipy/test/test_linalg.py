# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import numpy as np
import scipy
import jax
from jax import numpy as jnp
from jax import scipy as jsp
from pyscfad.scipy.linalg import eigh, svd, logm

def test_eigh():
    a = jnp.ones((2,2))
    b = jnp.eye(2)

    w0, v0 = jsp.linalg.eigh(a)
    w1, v1 = eigh(a)

    assert abs(w1-w0).max() < 1e-10
    assert abs(v1-v0).max() < 1e-10

    jac0 = jax.jacfwd(jsp.linalg.eigh)(a)
    jac1 = jax.jacfwd(eigh)(a)

    assert abs(jac1[0] - jac0[0]).max() < 1e-10
    assert abs(jac1[1] - jac0[1]).max() < 1e-10

    #finite difference
    disp = 0.0005
    b_m = jnp.array([[1., 0.],[0., 1.-disp]])
    w_m, v_m = eigh(a, b_m)
    b_p = jnp.array([[1., 0.],[0., 1.+disp]])
    w_p, v_p = eigh(a, b_p)

    g0 = (-v_p - v_m) / 0.001 # -v_p due to specific gauge

    jac = jax.jacfwd(eigh, argnums=1)(a, b)
    g1 = jac[1][...,1,1]
    assert abs(g1-g0).max() < 1e-7

def test_svd():
    a = jnp.ones((2,2))
    jac0 = jax.jacfwd(partial(jsp.linalg.svd, full_matrices=False))(a)
    jac1 = jax.jacfwd(svd)(a)
    assert abs(jac0[0] - jac1[0]).max() < 1e-7
    assert abs(jac0[1] - jac1[1]).max() < 1e-7
    assert abs(jac0[2] - jac1[2]).max() < 1e-7

def test_logm():
    theta = 2 * np.pi / 3
    a = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    A = scipy.linalg.block_diag(np.diag(np.array([-1, -1, 1])), a)

    X = logm(A, real=True)
    assert np.isreal(X).all()
    assert scipy.linalg.norm(scipy.linalg.expm(X) - A, 1) < 1e-12

    assert np.allclose(logm(A), scipy.linalg.logm(A))

