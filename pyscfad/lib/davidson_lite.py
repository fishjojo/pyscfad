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

"""
Implicitly-differentiable Davidson eigensolvers.

This module provides Davidson-iteration eigensolvers that are fully ``jax``
jittable and differentiable to arbitrary order, for both forward- and
reverse-mode. Instead of differentiating through the iterations, the converged
solution is differentiated via the implicit function theorem using
:func:`jax.lax.custom_root` (mirroring :mod:`pyscfad.scf.hf_lite`).

Two solvers are provided:

- :func:`eigh` for Hermitian (or real-symmetric) operators.
- :func:`eig`  for general (non-Hermitian) operators; the adjoint requires the
  transposed operator ``aopT`` (i.e. :math:`A^{H}`).

For (near-)degenerate eigenstates the response between states within
``deg_thresh`` is projected out, following the masking strategy in
:func:`pyscfad.backend._jax.lax.linalg._eigh_gen_jvp_rule`.

The forward solve uses a fixed-maximum-space, GPU-friendly Davidson iteration
(static shapes, batched ``aop``, thick restart via ``dynamic_update_slice``).
The number of returned roots is fixed (so the solver stays jittable); callers
must request enough roots to cover any degeneracy at the boundary.
"""
from __future__ import annotations

import jax
from jax import numpy as jnp
from jax import lax
from jax.lax import custom_root, while_loop, dynamic_update_slice

# Use the project gmres workaround for https://github.com/jax-ml/jax/issues/33872
# (jax's own gmres is not differentiable through custom_linear_solve here).
from pyscfad.scipy.sparse.linalg import gmres_const_atol

# Shift applied to the inactive (unfilled) part of the projected matrix so that
# the spurious roots are pushed above the physical spectrum and never selected
# among the lowest ``nroots``.
_BIG = 1e30


def _make_precond(adiag, precond):
    """Build a diagonal (Davidson) preconditioner from ``adiag`` if needed."""
    if precond is not None:
        return precond
    if adiag is not None:
        adiag = jnp.asarray(adiag).ravel()
        def _p(R, theta):
            denom = theta[None, :] - adiag[:, None]
            denom = jnp.where(jnp.abs(denom) < 1e-8, 1e-8, denom)
            return R / denom
        return _p
    return lambda R, theta: R


def _init_guess(x0, adiag, nroots, n, dtype):
    """Return an orthonormal ``(n, nroots)`` initial guess (jittable)."""
    if x0 is None:
        if adiag is None:
            raise ValueError('Either `x0` or `adiag` must be provided.')
        d = jax.lax.stop_gradient(jnp.asarray(adiag).real.ravel())
        idx = jnp.argsort(d)[:nroots]
        x0 = jax.nn.one_hot(idx, n, dtype=dtype).T
    else:
        x0 = jnp.asarray(x0)
        if x0.ndim == 1:
            x0 = x0[:, None]
        x0 = x0.astype(dtype)
    if x0.shape[1] != nroots:
        raise ValueError(f'guess has {x0.shape[1]} vectors, nroots={nroots}')
    return jnp.linalg.qr(x0)[0]


def _infer_dtype(aop, guess, force_complex=False):
    """Promote ``guess`` to the working dtype implied by ``aop`` (e.g. a complex
    Hermitian operator applied to a real guess yields a complex subspace)."""
    out = jax.eval_shape(aop, guess)
    dtype = jnp.result_type(guess.dtype, out.dtype)
    if force_complex:
        dtype = jnp.result_type(dtype, jnp.complex64)
    return guess.astype(dtype), dtype


def _orthonormalize(T, V, valid, m):
    """Orthonormalize columns of ``T`` against the first ``valid`` columns of ``V``."""
    mask = (jnp.arange(m) < valid)[None, :]
    Vm = V * mask
    T = T - Vm @ (Vm.conj().T @ T)
    T = T - Vm @ (Vm.conj().T @ T)
    return jnp.linalg.qr(T)[0]


def _davidson_solve(aop, x0, k, precond, tol, max_cycle, m, hermitian):
    """Fixed-max-space Davidson forward solve for the lowest ``k`` roots.

    Returns ``(theta, X)`` for the Hermitian case and ``(theta, X, Y)`` for the
    non-Hermitian case (``Y`` are the biorthonormal left vectors).
    """
    n = x0.shape[0]
    dtype = x0.dtype
    rdtype = jnp.zeros(1, dtype=dtype).real.dtype
    idx = jnp.arange(m)

    V0 = jnp.zeros((n, m), dtype=dtype).at[:, :k].set(x0)
    wdtype = rdtype if hermitian else dtype
    theta0 = jnp.zeros(k, dtype=wdtype)
    X0 = jnp.zeros((n, k), dtype=dtype)
    Y0 = jnp.zeros((n, k), dtype=dtype)
    init = (jnp.int32(0), V0, jnp.int32(k), jnp.bool_(False), theta0, X0, Y0)

    def cond(s):
        return (s[0] < max_cycle) & (~s[3])

    def body(s):
        icyc, V, nv, _, _, _, _ = s
        AV = aop(V)
        H = V.conj().T @ AV
        act = idx < nv
        actmat = act[:, None] & act[None, :]
        H = jnp.where(actmat, H, 0.0)
        H = H + jnp.diag(jnp.where(act, 0.0, _BIG)).astype(H.dtype)

        if hermitian:
            H = (H + H.conj().T) * 0.5
            w_all, C_all = jnp.linalg.eigh(H)
            order = jnp.arange(k)
            theta = w_all[order].astype(wdtype)
            C = C_all[:, order]
            Y = None
        else:
            w_all, C_all = jnp.linalg.eig(H)
            order = jnp.argsort(w_all.real)[:k]
            theta = w_all[order]
            C = C_all[:, order]
            # biorthonormal left vectors: rows of inv(C_all)
            Cinv = jnp.linalg.inv(C_all)
            L = Cinv[order, :].conj().T  # (m, k)

        X = V @ C
        AX = AV @ C
        R = AX - X * theta[None, :]
        conv = jnp.all(jnp.linalg.norm(R, axis=0) < tol)

        Yout = (V @ L) if not hermitian else Y0

        T = precond(R, theta.astype(dtype))

        restart = (nv + k) > m
        Vb, off, valid = lax.cond(
            restart,
            lambda: (jnp.zeros_like(V).at[:, :k].set(X), jnp.int32(k), jnp.int32(k)),
            lambda: (V, nv, nv))
        T = _orthonormalize(T, Vb, valid, m)
        Vnew = dynamic_update_slice(Vb, T.astype(Vb.dtype), (jnp.int32(0), off.astype(jnp.int32)))
        return (icyc + 1, Vnew, off + k, conv, theta, X, Yout)

    _, _, _, _, theta, X, Y = while_loop(cond, body, init)
    if hermitian:
        return theta, X
    return theta, X, Y


def eigh(aop, x0=None, *, nroots=1, adiag=None, precond=None, tol=1e-9,
         max_cycle=50, max_space=None, deg_thresh=1e-9,
         tangent_tol=1e-8, tangent_maxiter=50):
    """Lowest ``nroots`` eigenpairs of a Hermitian operator, differentiably.

    Args:
        aop: Linear operator. ``aop(V)`` returns ``A @ V`` for ``V`` of shape
            ``(n, p)``. Must be differentiable in any parameters of interest.
        x0: Optional ``(n,)`` or ``(n, nroots)`` initial guess. If ``None``,
            ``adiag`` is used to build a unit-vector guess.
        nroots: Number of lowest eigenpairs to return.
        adiag: Optional ``(n,)`` diagonal of ``A`` for the preconditioner and
            initial guess.
        precond: Optional preconditioner ``precond(R, theta) -> T``.
        tol: Residual convergence tolerance for the forward solve.
        max_cycle: Maximum number of Davidson cycles.
        max_space: Maximum subspace size (default ``max(2*nroots, nroots+8)``).
        deg_thresh: Eigenvalues within this threshold are treated as degenerate;
            the response between them is projected out.
        tangent_tol, tangent_maxiter: GMRES settings for the implicit-diff solve.

    Returns:
        ``(w, X)`` with eigenvalues ``w`` of shape ``(nroots,)`` (real) and
        eigenvectors ``X`` of shape ``(n, nroots)``.
    """
    k = nroots
    if max_space is None:
        max_space = max(2 * k, k + 8)
    if max_space < 2 * k:
        raise ValueError('max_space must be at least 2*nroots.')
    precond = _make_precond(adiag, precond)

    ref = jnp.asarray(x0) if x0 is not None else jnp.asarray(adiag)
    n = ref.shape[0]
    guess = _init_guess(x0, adiag, k, n, ref.dtype)
    guess, dtype = _infer_dtype(aop, guess)
    rdtype = jnp.zeros(1, dtype=dtype).real.dtype
    state0 = (guess, jnp.zeros(k, dtype=rdtype))

    def solve(_f, x0state):
        theta, X = _davidson_solve(aop, x0state[0], k, precond, tol,
                                   max_cycle, max_space, hermitian=True)
        return (X, theta.astype(rdtype))

    def root_fn(state):
        X, w = state
        R = aop(X) - X * w[None, :]
        s = 0.5 * (jnp.sum(jnp.conj(X) * X, axis=0).real - 1.0)
        return (R, s)

    def tangent_solve(g, b):
        Bx, bw = b
        # Recover the converged eigenvectors from the linearization rather than
        # closing over them, so higher-order derivatives are correct.
        X = -g((jnp.zeros_like(Bx), jnp.ones_like(bw)))[0]
        w = jnp.sum(jnp.conj(X) * aop(X), axis=0).real
        dw = -jnp.sum(jnp.conj(X) * Bx, axis=0).real

        deg = jnp.abs(w[None, :] - w[:, None]) <= deg_thresh

        def proj(y, dc):
            # P_k y = y - sum_{j deg with k} x_j (x_j^H y)
            return y - X @ (dc.astype(X.dtype) * (X.conj().T @ y))

        def col(carry, inp):
            Bxk, wk, dc = inp
            def lin(z):
                zp = proj(z, dc)
                return proj(aop(zp[:, None])[:, 0] - wk * zp, dc)
            zeta, _ = gmres_const_atol(
                lin, proj(Bxk, dc), atol=tangent_tol,
                maxiter=tangent_maxiter, solve_method='batched')
            return carry, proj(zeta, dc)

        _, zeta = lax.scan(col, None, (Bx.T, w.astype(Bx.dtype), deg))
        dX = bw.astype(Bx.dtype)[None, :] * X + zeta.T
        return (dX, dw)

    X, w = custom_root(root_fn, state0, solve, tangent_solve, has_aux=False)
    return w, X


def eig(aop, aopT=None, x0=None, *, nroots=1, adiag=None, precond=None,
        tol=1e-9, max_cycle=50, max_space=None, deg_thresh=1e-9,
        tangent_tol=1e-8, tangent_maxiter=50, return_left=False):
    """Lowest ``nroots`` (by real part) eigenpairs of a general operator.

    Note:
        Lowest-``nroots`` targeting for a non-Hermitian operator relies on the
        operator being diagonally dominant (as for CI/TDDFT matrices), so that
        the diagonal preconditioner and unit-vector initial guess steer the
        iteration toward the lowest real-part roots. For strongly non-normal
        operators the converged pairs are still exact eigenpairs but may not be
        the lowest ones; supply a good ``x0``/``adiag`` in that case.

    Args:
        aop: Linear operator ``aop(V) -> A @ V``.
        aopT: Adjoint operator ``aopT(V) -> A^H @ V``. Required for derivatives
            (used to recover the left eigenvectors). May be ``None`` if only the
            forward (non-differentiated) result is needed.
        x0, nroots, adiag, precond, tol, max_cycle, max_space, deg_thresh,
        tangent_tol, tangent_maxiter: see :func:`eigh`.
        return_left: If ``True``, also return the left eigenvectors ``Y``.

    Returns:
        ``(w, X)`` (or ``(w, X, Y)`` if ``return_left``) with complex
        eigenvalues ``w`` of shape ``(nroots,)``, right eigenvectors ``X`` and
        (optionally) biorthonormal left eigenvectors ``Y`` of shape
        ``(n, nroots)`` satisfying ``Y^H X = I``.
    """
    k = nroots
    if max_space is None:
        max_space = max(2 * k, k + 8)
    if max_space < 2 * k:
        raise ValueError('max_space must be at least 2*nroots.')
    precond = _make_precond(adiag, precond)

    ref = jnp.asarray(x0) if x0 is not None else jnp.asarray(adiag)
    n = ref.shape[0]
    guess = _init_guess(x0, adiag, k, n, ref.dtype)
    guess, cdtype = _infer_dtype(aop, guess, force_complex=True)

    def forward(x0vecs):
        theta, X, Y = _davidson_solve(aop, x0vecs, k, precond, tol,
                                      max_cycle, max_space, hermitian=False)
        return theta, X, Y

    if aopT is None:
        theta, X, Y = forward(guess)
        if return_left:
            return theta, X, Y
        return theta, X

    state0 = (guess, jnp.zeros((n, k), dtype=cdtype), jnp.zeros(k, dtype=cdtype))

    def solve(_f, x0state):
        theta, X, Y = forward(x0state[0])
        return (X, Y, theta.astype(cdtype))

    def root_fn(state):
        X, Y, w = state
        R1 = aop(X) - X * w[None, :]
        R2 = aopT(Y) - Y * jnp.conj(w)[None, :]
        s = jnp.sum(jnp.conj(Y) * X, axis=0) - 1.0
        return (R1, R2, s)

    def tangent_solve(g, b):
        B1, B2, bs = b
        zX = jnp.zeros_like(B1)
        zY = jnp.zeros_like(B2)
        ones = jnp.ones_like(bs)
        # Recover right and left eigenvectors from the linearization.
        X = -g((zX, zY, ones))[0]
        Y = -g((zX, zY, ones))[1]
        # Reconstruct w via Rayleigh quotient w_k = y_k^H A x_k / (y_k^H x_k).
        yx = jnp.sum(jnp.conj(Y) * X, axis=0)
        w = jnp.sum(jnp.conj(Y) * aop(X), axis=0) / yx

        deg = jnp.abs(w[None, :] - w[:, None]) <= deg_thresh

        def projR(v, dc):
            # oblique projector removing the (degenerate) right subspace;
            # v is a single column of shape (n,)
            return v - X @ (dc.astype(X.dtype) * (jnp.conj(Y).T @ v / yx))

        def projL(v, dc):
            return v - Y @ (dc.astype(Y.dtype) * (jnp.conj(X).T @ v / jnp.conj(yx)))

        # dw_k = - y_k^H B1_k / (y_k^H x_k)
        dw = -jnp.sum(jnp.conj(Y) * B1, axis=0) / yx

        def colx(carry, inp):
            B1k, wk, dwk, dc = inp
            rhs = B1k + 0.0  # (A - w_k) dx = B1k + x_k dw_k  -> rhs after proj
            def lin(z):
                zp = projR(z, dc)
                return projR(aop(zp[:, None])[:, 0] - wk * zp, dc)
            rhs = projR(rhs, dc)
            zeta, _ = gmres_const_atol(
                lin, rhs, atol=tangent_tol,
                maxiter=tangent_maxiter, solve_method='batched')
            return carry, projR(zeta, dc)

        _, zetaX = lax.scan(colx, None, (B1.T, w, dw, deg))
        dX = zetaX.T

        def coly(carry, inp):
            B2k, wk, dc = inp
            def lin(z):
                zp = projL(z, dc)
                return projL(aopT(zp[:, None])[:, 0] - jnp.conj(wk) * zp, dc)
            rhs = projL(B2k, dc)
            zeta, _ = gmres_const_atol(
                lin, rhs, atol=tangent_tol,
                maxiter=tangent_maxiter, solve_method='batched')
            return carry, projL(zeta, dc)

        _, zetaY = lax.scan(coly, None, (B2.T, w, deg))
        dY = zetaY.T

        # Fix the gauge (scale) of dX, dY from the biorthonormality tangent bs:
        #   d(Y^H X) = bs  ->  diag(dY^H X + Y^H dX) = bs
        # split symmetrically between the X- and Y-gauge directions.
        gauge = bs - (jnp.sum(jnp.conj(dY) * X, axis=0)
                      + jnp.sum(jnp.conj(Y) * dX, axis=0))
        dX = dX + 0.5 * gauge[None, :] / jnp.conj(yx)[None, :] * X
        dY = dY + 0.5 * jnp.conj(gauge)[None, :] / yx[None, :] * Y
        return (dX, dY, dw)

    X, Y, w = custom_root(root_fn, state0, solve, tangent_solve, has_aux=False)
    if return_left:
        return w, X, Y
    return w, X
