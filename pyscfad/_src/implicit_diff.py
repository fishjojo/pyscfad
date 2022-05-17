import inspect
import jax
from jaxopt import linear_solve
from jaxopt.tree_util import tree_sub, tree_scalar_mul


def _map_back(diff_items, items, keys):
    new_items = list(items)
    for k, i in enumerate(keys):
        new_items[i] = diff_items[k]
    return tuple(new_items)

def root_vjp(optimality_fun, sol, args, cotangent,
             solve=linear_solve.solve_gmres, nondiff_argnums=()):

    def fun_sol(sol):
        return optimality_fun(sol, *args)

    _, vjp_fun_sol = jax.vjp(fun_sol, sol)

    matvec = lambda u: vjp_fun_sol(u)[0]
    v = tree_scalar_mul(-1, cotangent)
    u = solve(matvec, v)

    diff_args_dict = {i: arg for i, arg in enumerate(args) if i+1 not in nondiff_argnums}
    keys = diff_args_dict.keys()
    diff_args = tuple(diff_args_dict.values())
    def fun_args(*diff_args):
        new_args = _map_back(diff_args, args, keys)
        return optimality_fun(sol, *new_args)

    _, vjp_fun_args = jax.vjp(fun_args, *diff_args)
    diff_vjps = vjp_fun_args(u)

    vjps = [None,] * len(args)
    vjps = _map_back(diff_vjps, vjps, keys)
    return vjps

def _custom_root(solver_fun, optimality_fun, solve,
                 has_aux=False, nondiff_argnums=(), use_converged_args=None):
    solver_fun_sig = inspect.signature(solver_fun)
    optimality_fun_sig = inspect.signature(optimality_fun)

    def make_custom_vjp_solver_fun(solver_fun, kwargs):
        @jax.custom_vjp
        def solver_fun_close_kwargs(*args):
            return solver_fun(*args, **kwargs)

        def solver_fun_fwd(*args):
            res = solver_fun_close_kwargs(*args)
            return res, (res, args)

        def solver_fun_rev(tup, cotangent):
            res, args = tup
            if has_aux:
                cotangent = cotangent[0]
                sol = res[0]
            else:
                sol = res

            args = list(args)
            if use_converged_args:
                for key, val in use_converged_args.items():
                    args[key] = res[val]

            try:
                _ = optimality_fun_sig.bind(*args)
            except TypeError as err:
                msg = ("The optimality function has arguments that "
                       "are not compatible with the solver function.")
                raise TypeError(msg) from err

            vjps = root_vjp(optimality_fun, sol,
                            args[1:], cotangent, solve=solve,
                            nondiff_argnums=nondiff_argnums)
            vjps = (None,) + vjps
            return vjps

        solver_fun_close_kwargs.defvjp(solver_fun_fwd, solver_fun_rev)
        return solver_fun_close_kwargs

    def wrapped_solver_fun(*args, **kwargs):
        ba = solver_fun_sig.bind(*args, **kwargs)
        ba.apply_defaults()
        return make_custom_vjp_solver_fun(solver_fun, ba.kwargs)(*ba.args)

    return wrapped_solver_fun

def custom_root(optimality_fun, solve=None, has_aux=False,
                nondiff_argnums=(), use_converged_args=None):
    if solve is None:
        solve = linear_solve.solve_gmres

    def wrapper(solver_fun):
        return _custom_root(solver_fun, optimality_fun, solve,
                            has_aux=has_aux, nondiff_argnums=nondiff_argnums,
                            use_converged_args=use_converged_args)

    return wrapper

def custom_fixed_point(fixed_point_fun, solve=None, has_aux=False,
                       nondiff_argnums=(), use_converged_args=None):

    def optimality_fun(x0, *args):
        return tree_sub(fixed_point_fun(x0, *args), x0)

    optimality_fun.__wrapped__ = fixed_point_fun

    return custom_root(optimality_fun, solve=solve,
                       has_aux=has_aux, nondiff_argnums=nondiff_argnums,
                       use_converged_args=use_converged_args)
