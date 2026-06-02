# Copyright 2025-2026 The PySCFAD Authors
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

import numpy as np
import pytest
import jax
from jax import numpy as jnp
from jax.test_util import check_grads

from pyscfad.lib.davidson_lite import eigh, eig


def _sym(seed, n):
    r = np.random.default_rng(seed).standard_normal((n, n))
    return jnp.asarray(r + r.T)


def _gen(seed, n):
    r = np.random.default_rng(seed).standard_normal((n, n))
    return jnp.asarray(r)


def _csym(seed, n):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return jnp.asarray(r + r.conj().T)


def _ddom(seed, n, scale=0.3):
    # Diagonally dominant non-symmetric matrix with well-separated, ascending
    # real eigenvalues -- the regime where lowest-N Davidson targeting works
    # (mirrors CI/TDDFT matrices).
    rng = np.random.default_rng(seed)
    M = scale * rng.standard_normal((n, n)) + np.diag(np.arange(1.0, n + 1))
    return jnp.asarray(M)


def test_eigh_forward():
    n, k = 10, 3
    A = _sym(0, n)
    w, X = eigh(lambda Z: A @ Z, nroots=k, adiag=jnp.diag(A),
                max_space=2 * k + 4, tol=1e-11)
    ref = jnp.linalg.eigvalsh(A)[:k]
    assert float(abs(w - ref).max()) < 1e-8
    assert float(abs(A @ X - X * w[None, :]).max()) < 1e-6
    assert float(abs(X.conj().T @ X - jnp.eye(k)).max()) < 1e-9


def test_eigh_jit():
    n, k = 8, 2
    A = _sym(5, n)
    adiag = jnp.diag(A)
    fn = jax.jit(lambda M: eigh(lambda Z: M @ Z, nroots=k, adiag=adiag,
                                max_space=2 * k + 4, tol=1e-11)[0])
    w = fn(A)
    ref = jnp.linalg.eigvalsh(A)[:k]
    assert float(abs(w - ref).max()) < 1e-8


def test_eigh_grad_hellmann_feynman():
    n, k = 8, 2
    A0 = _sym(1, n)
    B = _sym(2, n)

    def f(p):
        A = A0 + p * B
        w, _ = eigh(lambda Z: A @ Z, nroots=k, adiag=jnp.diag(A),
                    max_space=2 * k + 6, tol=1e-12)
        return jnp.sum(w)

    _, V = jnp.linalg.eigh(A0)
    ref = sum(float(V[:, i] @ B @ V[:, i]) for i in range(k))
    g = float(jax.grad(f)(0.0))
    assert abs(g - ref) < 1e-6


def test_eigh_check_grads_order2():
    n, k = 6, 1
    A0 = _sym(3, n)
    B = _sym(4, n)

    def f(p):
        A = A0 + p * B
        return eigh(lambda Z: A @ Z, nroots=k, adiag=jnp.diag(A),
                    max_space=2 * k + 6, tol=1e-12)[0][0]

    check_grads(f, (0.3,), order=2, modes=['fwd', 'rev'], atol=1e-3, rtol=1e-3)


def test_eigh_grad_eigvec_vs_eigh():
    # Compare gradient of a smooth scalar of the lowest eigenvector against the
    # dense jnp.linalg.eigh path.
    n = 7
    A0 = _sym(11, n)
    B = _sym(12, n)
    c = jnp.asarray(np.random.default_rng(13).standard_normal(n))

    def obj_davidson(p):
        A = A0 + p * B
        _, X = eigh(lambda Z: A @ Z, nroots=1, adiag=jnp.diag(A),
                    max_space=8, tol=1e-12)
        x = X[:, 0]
        x = x * jnp.sign(x[0])
        return jnp.sum((c * x) ** 2)

    def obj_dense(p):
        A = A0 + p * B
        w, V = jnp.linalg.eigh(A)
        x = V[:, 0]
        x = x * jnp.sign(x[0])
        return jnp.sum((c * x) ** 2)

    g1 = float(jax.grad(obj_davidson)(0.2))
    g2 = float(jax.grad(obj_dense)(0.2))
    assert abs(g1 - g2) < 1e-5


def test_eigh_degenerate():
    # Block-diagonal with a doubly-degenerate lowest eigenvalue. The eigenvalue
    # derivative and the (gauge-invariant) projector derivative must be finite.
    n = 6
    rng = np.random.default_rng(21)
    Q = jnp.asarray(np.linalg.qr(rng.standard_normal((n, n)))[0])
    evals = jnp.asarray([1.0, 1.0, 3.0, 4.0, 5.0, 6.0])
    A0 = (Q * evals[None, :]) @ Q.T
    B = _sym(22, n)

    def proj_obj(p):
        A = A0 + p * B
        _, X = eigh(lambda Z: A @ Z, nroots=2, adiag=jnp.diag(A),
                    max_space=8, tol=1e-11)
        P = X @ X.conj().T
        return jnp.vdot(P, P).real

    val, g = jax.value_and_grad(proj_obj)(0.0)
    assert np.isfinite(val)
    assert np.isfinite(g)


def test_eigh_complex_hermitian():
    n, k = 6, 2
    H0 = _csym(60, n)
    B = _csym(61, n)
    w, X = eigh(lambda Z: H0 @ Z, nroots=k, adiag=jnp.diag(H0).real,
                max_space=8, tol=1e-12)
    assert float(abs(w - jnp.linalg.eigvalsh(H0)[:k]).max()) < 1e-8
    assert float(abs(X.conj().T @ X - jnp.eye(k)).max()) < 1e-9

    def f(p):
        H = H0 + p * B
        ww, _ = eigh(lambda Z: H @ Z, nroots=k, adiag=jnp.diag(H).real,
                     max_space=8, tol=1e-12)
        return jnp.sum(ww)

    _, V = jnp.linalg.eigh(H0)
    ref = sum(float((V[:, i].conj() @ B @ V[:, i]).real) for i in range(k))
    assert abs(float(jax.grad(f)(0.0)) - ref) < 1e-6


def test_eig_forward():
    n, k = 12, 3
    A = _ddom(30, n)
    w, X, Y = eig(lambda Z: A @ Z, aopT=lambda Z: A.T @ Z, nroots=k,
                  adiag=jnp.diag(A), max_space=2 * k + 6, tol=1e-10,
                  return_left=True)
    wref = np.linalg.eigvals(np.asarray(A))
    wref = wref[np.argsort(wref.real)][:k]
    diff = np.abs(np.sort_complex(np.asarray(w)) - np.sort_complex(wref)).max()
    assert diff < 1e-6
    assert float(abs(A @ X - X * w[None, :]).max()) < 1e-5
    # biorthonormal left vectors: Y^H X = I
    assert float(abs(Y.conj().T @ X - jnp.eye(k)).max()) < 1e-8


def test_eig_jit():
    n, k = 10, 2
    A = _ddom(31, n)
    adiag = jnp.diag(A)
    fn = jax.jit(lambda M: eig(lambda Z: M @ Z, aopT=lambda Z: M.T @ Z,
                               nroots=k, adiag=adiag, max_space=12, tol=1e-10)[0])
    w = fn(A)
    wref = np.linalg.eigvals(np.asarray(A))
    wref = wref[np.argsort(wref.real)][:k]
    diff = np.abs(np.sort_complex(np.asarray(w)) - np.sort_complex(wref)).max()
    assert diff < 1e-6


def test_eig_restart_orthonormal():
    # Large diagonally dominant matrix with the (small) default max_space so the
    # solve must thick-restart several times. The restart must re-orthonormalize
    # the non-Hermitian Ritz vectors; otherwise it selects spurious near-zero
    # Ritz pairs (eigenvalues near 0 with tiny vectors) instead of the lowest
    # roots near 1, 2. See PR #119 review.
    n, k = 80, 2
    A = _ddom(50, n)
    w, X, Y = eig(lambda Z: A @ Z, aopT=lambda Z: A.T @ Z, nroots=k,
                  adiag=jnp.diag(A), tol=1e-10, return_left=True)
    wref = np.linalg.eigvals(np.asarray(A))
    wref = wref[np.argsort(wref.real)][:k]
    diff = np.abs(np.sort_complex(np.asarray(w)) - np.sort_complex(wref)).max()
    assert diff < 1e-6
    # exact eigenpairs with non-degenerate (O(1)) right vectors
    assert float(abs(A @ X - X * w[None, :]).max()) < 1e-5
    assert float(jnp.linalg.norm(X, axis=0).min()) > 1e-3
    assert float(abs(Y.conj().T @ X - jnp.eye(k)).max()) < 1e-8


def test_eig_check_grads_order2():
    # Non-Hermitian eigenvalue gradient via aopT, forward and reverse, 2nd order.
    n, k = 8, 1
    A0 = _ddom(40, n)
    B = _ddom(41, n)

    def f(p):
        A = A0 + p * B
        w, _ = eig(lambda Z: A @ Z, aopT=lambda Z: A.T @ Z, nroots=k,
                   adiag=jnp.diag(A), max_space=8, tol=1e-12)
        return w[0].real

    check_grads(f, (0.05,), order=2, modes=['fwd', 'rev'], atol=2e-3, rtol=2e-3)


def test_eig_eigvec_grad_stopped():
    # The non-Hermitian eigenvectors have an undetermined complex scaling gauge
    # that the forward (jnp.linalg.eig) normalization pins non-smoothly, so eig
    # stops their gradients while keeping the (gauge-invariant) eigenvalue
    # differentiable. Confirm: d/dp of a scalar of X (and Y) is exactly zero,
    # but d/dp of the eigenvalue is finite and matches finite differences.
    n, k = 8, 1
    A0 = _ddom(40, n)
    B = _ddom(41, n)
    c = jnp.asarray(np.random.default_rng(70).standard_normal(n))

    def vec_obj(p):
        A = A0 + p * B
        _, X, Y = eig(lambda Z: A @ Z, aopT=lambda Z: A.T @ Z, nroots=k,
                      adiag=jnp.diag(A), max_space=8, tol=1e-12,
                      return_left=True)
        return jnp.sum((c * X[:, 0].real) ** 2) + jnp.sum((c * Y[:, 0].real) ** 2)

    assert float(jax.grad(vec_obj)(0.05)) == 0.0

    def val_obj(p):
        A = A0 + p * B
        w, _ = eig(lambda Z: A @ Z, aopT=lambda Z: A.T @ Z, nroots=k,
                   adiag=jnp.diag(A), max_space=8, tol=1e-12)
        return w[0].real

    g = float(jax.grad(val_obj)(0.05))
    h = 1e-4
    fd = float((val_obj(0.05 + h) - val_obj(0.05 - h)) / (2 * h))
    assert abs(g) > 1e-6
    assert abs(g - fd) < 1e-5
