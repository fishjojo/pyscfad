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

from typing import NamedTuple, Any
import operator
import jax
from jax import tree
from pyscfad import numpy as np
from pyscfad import pytree

Array = Any

def _tree_scale_sum(a, xs):
    return tree.map(lambda x: np.tensordot(a, x, axes=1), xs)

def _tree_axpy(a, xs, ys):
    return tree.map(lambda x, y: a * x + y, xs, ys)

def _tree_sub(xs, ys):
    return tree.map(operator.sub, xs, ys)

def _tree_set(xs, idx, vals):
    return tree.map(lambda x, val: x.at[idx].set(val), xs, vals)

def _tree_vdot(xs, ys):
    vdots = tree.map(np.vdot, xs, ys)
    return tree.reduce(operator.add, vdots)

def minimize_residual(
    res_gram: Array,
    ridge: float,
) -> Array:
    m = res_gram.shape[0]
    res_gram = res_gram + ridge * np.eye(m) # avoid divergence
    H = np.block([[np.zeros((1, 1)), np.ones((1, m))],
                  [ np.ones((m, 1)), res_gram       ]])
    c = np.zeros((m+1)).at[0].set(1)
    alpha = np.linalg.solve(H, c)
    return alpha

def anderson_step(
    param_hist: Any,
    res_hist: Any,
    res_gram: Array,
    ridge: float,
    beta: float,
) -> Any:
    alpha = minimize_residual(res_gram, ridge)
    alpha = alpha[1:]
    ax = _tree_scale_sum(alpha, param_hist)
    aw = _tree_scale_sum(alpha, res_hist)
    param = _tree_axpy(beta, aw, ax)
    return param

def update_history(
    pos: int,
    param_hist: Any,
    res_hist: Any,
    res_gram: Array,
    param: Any,
    residual: Any,
) -> tuple[Any, Any, Array]:
    param_hist = _tree_set(param_hist, pos, param)
    res_hist = _tree_set(res_hist, pos, residual)
    new_row = jax.vmap(_tree_vdot, in_axes=(0, None))(res_hist, residual)
    res_gram = res_gram.at[pos,:].set(new_row)
    res_gram = res_gram.at[:,pos].set(new_row)
    return param_hist, res_hist, res_gram

class AndersonState(NamedTuple):
    cycle: int
    param_hist: Any
    res_hist: Any
    res_gram: Array

class Anderson(pytree.PytreeNode):
    r"""Anderson mixing.

    Parameters
    ----------
    param: parameters to be mixed.
    space: size of subspace.
    ridge: ridge regularization.
    damp: damping factor; :math:`1-\beta` in Eq. 2.3.
    start_cycle: starting cycle for mixing.

    References
    ----------
    Pollock, Sara, and Leo G. Rebholz.
    "Anderson acceleration for contractive and noncontractive operators."
    IMA Journal of Numerical Analysis 41.4 (2021): 2841-2872.
    """
    _dynamic_attr = ["state"]

    def __init__(
        self,
        param: Any,
        space: int = 6,
        ridge: float = 1e-10,
        damp: float = 0,
        start_cycle: int = 1,
    ):
        self.space = space
        self.ridge = ridge
        self.damp = damp
        self.start_cycle = start_cycle
        self.state = self.init_state(param)

    def init_state(
        self,
        param: Any,
    ) -> NamedTuple:
        m = self.space
        param_hist = tree.map(lambda x: np.tile(x, [m]+[1]*x.ndim), param)
        res_hist = tree.map(np.zeros_like, param_hist)
        res_gram = np.zeros((m,m))
        state = AndersonState(
            cycle=np.asarray(0, dtype=int),
            param_hist=param_hist,
            res_hist=res_hist,
            res_gram=res_gram,
        )
        return state

    def update(
        self,
        param: Any,
        param_last: Any,
    ) -> Any:
        state = self.state
        cycle = state.cycle
        param_hist = state.param_hist
        res_hist = state.res_hist
        res_gram = state.res_gram

        pos = np.mod(cycle, self.space)
        residual = _tree_sub(param, param_last)
        param_hist, res_hist, res_gram = \
            update_history(pos, param_hist, res_hist, res_gram, param_last, residual)

        next_state = AndersonState(
            cycle=cycle+1,
            param_hist=param_hist,
            res_hist=res_hist,
            res_gram=res_gram,
        )

        def _anderson_step(param, state):
            extrapolated = anderson_step(
                state.param_hist,
                state.res_hist,
                state.res_gram,
                self.ridge,
                1.-self.damp,
            )
            return extrapolated

        def _use_param(param, state):
            return param * (1.-self.damp) + param_last * self.damp

        start_cycle = jax.lax.select(
            np.greater_equal(self.start_cycle+1, self.space),
            self.start_cycle+1,
            self.space,
        )
        #start_cycle = np.minimum(self.start_cycle+1, self.space)
        extrapolated = jax.lax.cond(
            np.greater_equal(cycle+1, start_cycle),
            _anderson_step,
            _use_param,
            param,
            next_state,
        )

        self.state = next_state
        return extrapolated
