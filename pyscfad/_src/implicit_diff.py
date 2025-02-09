import inspect
import operator
from functools import partial
import jax
from jax.tree_util import tree_map
from jax.scipy.sparse.linalg import gmres

_Sub = partial(tree_map, operator.sub)

def _Scalar_mul(scal, tree_x):
    return tree_map(lambda x: scal * x, tree_x)

def _map_back(diff_items, items, keys):
    new_items = list(items)
    for k, i in enumerate(keys):
        new_items[i] = diff_items[k]
    return tuple(new_items)

def root_vjp(optimality_fun, sol, args, cotangent,
             solve=gmres, nondiff_argnums=(),
             optfn_has_aux=False, solver_kwargs=None,
             gen_precond=None,
             custom_vjp_from_optcond=False):
    if solver_kwargs is None:
        solver_kwargs = {}

    def fun_sol(sol):
        return optimality_fun(sol, *args)

    # FIXME M may not work for solvers other than scipy
    M = None
    if optfn_has_aux:
        if custom_vjp_from_optcond:
            _, (vjp_fun_sol, optfn_aux) = fun_sol(sol)
        else:
            _, vjp_fun_sol, optfn_aux = jax.vjp(fun_sol, sol, has_aux=True)
        if gen_precond is not None:
            M = gen_precond(optfn_aux)
    else:
        _, vjp_fun_sol = jax.vjp(fun_sol, sol)

    def matvec(u):
        return vjp_fun_sol(u)[0]

    v = _Scalar_mul(-1, cotangent)
    u = solve(matvec, v, M=M, **solver_kwargs)[0]

    diff_args_dict = {i: arg for i, arg in enumerate(args) if i+1 not in nondiff_argnums}
    keys = diff_args_dict.keys()
    diff_args = tuple(diff_args_dict.values())
    def fun_args(*diff_args):
        new_args = _map_back(diff_args, args, keys)
        return optimality_fun(sol, *new_args)

    _, vjp_fun_args = jax.vjp(fun_args, *diff_args, has_aux=optfn_has_aux)[:2]
    diff_vjps = vjp_fun_args(u)

    vjps = [None,] * len(args)
    vjps = _map_back(diff_vjps, vjps, keys)
    return vjps

def _custom_root(solver_fun, optimality_fun, solve,
                 has_aux=False, nondiff_argnums=(), use_converged_args=None,
                 optfn_has_aux=False, solver_kwargs=None,
                 gen_precond=None,
                 custom_vjp_from_optcond=False):
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
                msg = ('The optimality function has arguments that '
                       'are not compatible with the solver function.')
                raise TypeError(msg) from err

            vjps = root_vjp(optimality_fun, sol,
                            args[1:], cotangent, solve=solve,
                            nondiff_argnums=nondiff_argnums,
                            optfn_has_aux=optfn_has_aux,
                            solver_kwargs=solver_kwargs,
                            gen_precond=gen_precond,
                            custom_vjp_from_optcond=custom_vjp_from_optcond)
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
                nondiff_argnums=(), use_converged_args=None,
                optfn_has_aux=False, solver_kwargs=None,
                gen_precond=None,
                custom_vjp_from_optcond=False):
    if solve is None:
        solve = gmres

    def wrapper(solver_fun):
        return _custom_root(solver_fun, optimality_fun, solve,
                            has_aux=has_aux, nondiff_argnums=nondiff_argnums,
                            use_converged_args=use_converged_args,
                            optfn_has_aux=optfn_has_aux,
                            solver_kwargs=solver_kwargs,
                            gen_precond=gen_precond,
                            custom_vjp_from_optcond=custom_vjp_from_optcond)

    return wrapper

def custom_fixed_point(fixed_point_fun, solve=None, has_aux=False,
                       nondiff_argnums=(), use_converged_args=None,
                       optfn_has_aux=False, solver_kwargs=None,
                       gen_precond=None,
                       custom_vjp_from_optcond=False):

    def optimality_fun(x0, *args):
        return _Sub(fixed_point_fun(x0, *args), x0)

    optimality_fun.__wrapped__ = fixed_point_fun

    return custom_root(optimality_fun, solve=solve,
                       has_aux=has_aux, nondiff_argnums=nondiff_argnums,
                       use_converged_args=use_converged_args,
                       optfn_has_aux=optfn_has_aux,
                       solver_kwargs=solver_kwargs,
                       gen_precond=gen_precond,
                       custom_vjp_from_optcond=custom_vjp_from_optcond)

def make_implicit_diff(fn, implicit_diff=False, fixed_point=True,
                       optimality_cond=None, solver=None, has_aux=False,
                       nondiff_argnums=(), use_converged_args=None,
                       optimality_fun_has_aux=False,
                       solver_kwargs=None, gen_precond=None,
                       custom_vjp_from_optcond=False):
    """Wrap a function for implicit differentiation.

    Parameters
    ----------
    fn : object
        The function to be wrapped.
    implicit_diff : bool, optional
        If False, return `fn` without modification.
        Otherwise, wrap `fn` to make it implicitly differentiable.
        Default is ``False``.
    fixed_point : bool, optional
        If True, the optimality condition is defined by a fixed point
        problem. Otherwise, the optimality condition should return 0.
        Default is ``True``.
    optimality_cond : object, optional
        The funtion defining the optimality condition problem.
    solver : object, optional
        The function solves the linear equations.
    has_aux : bool, optional
        Whether `fn` returns auxiliary data. Default is ``False``.
    nondiff_argnums : tuple of ints, optional
        Specify which arguments are not differentiated with respect to,
        by their indices in the argument list. Default is empty tuple.
    use_converged_args : dict, optional
        Specify in the backward propagation, which arguments should use
        their converged values produced by `fn` in the forward pass.
        The dict has the form ``{argnum: value}``, where ``argnum``
        is the index of this argument in the argument list. Default is ``None``.
    optimality_fun_has_aux : bool, optional
        Whether the optimality function returns auxiliary data.
        Default is False.
    solver_kwargs : dict, optional
        The keyword arguments passed to the linear solver function.
        Default is empty dict.
    gen_precond : object, optional
        A function which generates the preconditioner
        for the linear solver. `gen_precond` should
        take the auxiliary data returned by the optimality function as
        its argument, and return the preconditioner recognized by the
        linear solver, which is usually a function or an array.

    Returns
    -------
    wrapped_fn : object
        The wrapped function.
    """
    if implicit_diff:
        if fixed_point:
            method = custom_fixed_point
        else:
            method = custom_root

        if not callable(optimality_cond):
            raise KeyError(f'optimality_cond must be a function, '
                           f'but get{optimality_cond}.')
        if solver is None:
            solver = gmres
        return method(optimality_cond, solve=solver, has_aux=has_aux,
                      nondiff_argnums=nondiff_argnums,
                      use_converged_args=use_converged_args,
                      optfn_has_aux=optimality_fun_has_aux,
                      solver_kwargs=solver_kwargs,
                      gen_precond=gen_precond,
                      custom_vjp_from_optcond=custom_vjp_from_optcond)(fn)
    else:
        return fn
