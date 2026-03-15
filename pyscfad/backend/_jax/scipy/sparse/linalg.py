# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from jax._src import api
from jax._src import lax
from jax._src import numpy as jnp
from jax._src.tree_util import (
  tree_leaves,
  tree_map,
  tree_structure,
)
from jax._src.scipy.sparse.linalg import (
  _identity,
  _normalize_matvec,
  _norm,
  _gmres_incremental,
  _gmres_batched,
  _gmres_solve,
)

def gmres(A, b, x0=None, *, tol=1e-5, atol=1e-5, restart=20, maxiter=None,
          M=None, solve_method='batched'):
  """A workaround for https://github.com/jax-ml/jax/issues/33872.
  Currently, ``atol`` and ``ptol`` can not depend on ``b``,
  so they are set as constants.
  """
  if x0 is None:
    x0 = tree_map(jnp.zeros_like, b)
  if M is None:
    M = _identity
  A = _normalize_matvec(A)
  M = _normalize_matvec(M)

  b = api.device_put(b)
  x0 = api.device_put(x0)
  size = sum(bi.size for bi in tree_leaves(b))

  if maxiter is None:
    maxiter = 10 * size  # copied from scipy
  restart = min(restart, size)

  if tree_structure(x0) != tree_structure(b):
    raise ValueError(
        'x0 and b must have matching tree structure: '
        f'{tree_structure(x0)} vs {tree_structure(b)}')

  #b_norm = _norm(b)
  #atol = jnp.maximum(tol * b_norm, atol)

  #Mb = M(b)
  #Mb_norm = _norm(Mb)
  #ptol = Mb_norm * jnp.minimum(1.0, atol / b_norm)
  ptol = atol

  if solve_method == 'incremental':
    gmres_func = _gmres_incremental
  elif solve_method == 'batched':
    gmres_func = _gmres_batched
  else:
    raise ValueError(f"invalid solve_method {solve_method}, must be either "
                     "'incremental' or 'batched'")

  def _solve(A, b):
    return _gmres_solve(A, b, x0, atol, ptol, restart, maxiter, M, gmres_func)
  x = lax.custom_linear_solve(A, b, solve=_solve, transpose_solve=_solve)

  failed = jnp.isnan(_norm(x))
  info = jnp.where(failed, -1, 0)
  return x, info
