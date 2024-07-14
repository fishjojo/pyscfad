"""
Custom jax.lax.linalg functions
"""
from functools import partial
import numpy as np
import jax
from jax import lax
from jax._src import ad_util
from jax._src import api
from jax._src import dispatch
from jax._src.core import (
    Primitive,
    ShapedArray,
    is_constant_shape,
)
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import lax as lax_internal
from jax._src.lax.linalg import (
    _H,
    symmetrize,
    _nan_like_hlo,
    _broadcasting_select_hlo,
)
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy import ufuncs

from pyscfadlib import lapack as lp

def eigh_gen(a, b, *,
             lower=True,
             itype=1,
             deg_thresh=1e-9):
    a = symmetrize(a)
    b = symmetrize(b)
    w, v = eigh_gen_p.bind(a, b, lower=lower, itype=itype, deg_thresh=deg_thresh)
    return w, v

def _eigh_gen_impl(a, b, *, lower, itype, deg_thresh):
    w, v = dispatch.apply_primitive(
                eigh_gen_p,
                a, b,
                lower=lower,
                itype=itype,
                deg_thresh=deg_thresh)
    return w, v

def _eigh_gen_abstract_eval(a, b, *, lower, itype, deg_thresh):
    if isinstance(a, ShapedArray):
        if a.ndim < 2 or a.shape[-2] != a.shape[-1]:
            raise ValueError(
                "Argument \'a\' to eigh must have shape [..., n, n], "
                "but got shape {}".format(a.shape))

        batch_dims = a.shape[:-2]
        n = a.shape[-1]
        v = a.update(shape=batch_dims + (n, n))
        w = a.update(
                shape=batch_dims + (n,),
                dtype=lax_internal._complex_basetype(a.dtype))
    else:
        w, v = a, a
    return w, v

def _eigh_gen_jvp_rule(primals, tangents, *, lower, itype, deg_thresh):
    if itype != 1:
        raise NotImplementedError(f"JVP for itype={itype} is not implemented.")
    a, b = primals
    n = a.shape[-1]
    at, bt = tangents

    w_real, v = eigh_gen_p.bind(
                    symmetrize(a),
                    symmetrize(b),
                    lower=lower,
                    itype=itype,
                    deg_thresh=deg_thresh)

    w = w_real.astype(a.dtype)
    eji = w[..., jnp.newaxis, :] - w[..., jnp.newaxis]
    Fmat = ufuncs.reciprocal(
        jnp.where(ufuncs.absolute(eji) > deg_thresh, eji, jnp.inf)
    )

    dot = partial(lax.dot if a.ndim == 2 else lax.batch_matmul,
                  precision=lax.Precision.HIGHEST)

    if type(at) is ad_util.Zero:
        vt_at_v = lax.zeros_like_array(a)
    else:
        vt_at_v = dot(_H(v), dot(at, v))

    if type(bt) is not ad_util.Zero:
        if a.ndim == 2:
            w_diag = jnp.diag(w)
        else:
            batch_dims = a.shape[:-2]
            w_diag = api.vmap(jnp.diag, in_axes=batch_dims, out_axes=batch_dims)(w)
        vt_bt_v = dot(_H(v), dot(bt, v))
        vt_bt_v_w = dot(vt_bt_v, w_diag)
        vt_at_v -= vt_bt_v_w

    dw = ufuncs.real(jnp.diagonal(vt_at_v, axis1=-2, axis2=-1))

    F_vt_at_v = ufuncs.multiply(Fmat, vt_at_v)
    if type(bt) is not ad_util.Zero:
        bmask = jnp.where(ufuncs.absolute(eji) > deg_thresh, jnp.zeros_like(a), 1)
        F_vt_at_v -= ufuncs.multiply(bmask, vt_bt_v) * .5

    dv = dot(v, F_vt_at_v)
    return (w_real, v), (dw, dv)

def _eigh_gen_cpu_lowering(ctx, a, b, *, lower, itype, deg_thresh):
    del deg_thresh
    a_aval, b_aval = ctx.avals_in
    w_aval, v_aval = ctx.avals_out
    n = a_aval.shape[-1]
    batch_dims = a_aval.shape[:-2]

    if not is_constant_shape(a_aval.shape[-2:]):
        raise NotImplementedError(
            "Shape polymorphism for native lowering for eigh is implemented "
            f"only for the batch dimensions: {a_aval.shape}")

    a_shape_vals = mlir.eval_dynamic_shape_as_ivals(ctx, a_aval.shape)
    b_shape_vals = mlir.eval_dynamic_shape_as_ivals(ctx, b_aval.shape)
    v, w, info = lp.sygvd_hlo(a_aval.dtype, a, b,
                              a_shape_vals=a_shape_vals,
                              b_shape_vals=b_shape_vals,
                              lower=lower, itype=itype)

    zeros = mlir.full_like_aval(ctx, 0, ShapedArray(batch_dims, np.dtype(np.int32)))
    ok = mlir.compare_hlo(info, zeros, "EQ", "SIGNED")
    select_v_aval = ShapedArray(batch_dims + (1, 1), np.dtype(np.bool_))
    v = _broadcasting_select_hlo(
            ctx,
            mlir.broadcast_in_dim(ctx, ok, select_v_aval,
                                  broadcast_dimensions=range(len(batch_dims))),
            select_v_aval,
            v, v_aval, _nan_like_hlo(ctx, v_aval), v_aval)
    select_w_aval = ShapedArray(batch_dims + (1,), np.dtype(np.bool_))
    w = _broadcasting_select_hlo(
        ctx,
        mlir.broadcast_in_dim(ctx, ok, select_w_aval,
                              broadcast_dimensions=range(len(batch_dims))),
        select_w_aval,
        w, w_aval, _nan_like_hlo(ctx, w_aval), w_aval)
    return [w, v]

def _eigh_gen_batching_rule(batched_args, batch_dims, *,
                            lower, itype, deg_thresh):
    a, b = batched_args
    bd_a, bd_b = batch_dims
    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims)
                if i is not None)
    a = batching.bdim_at_front(a, bd_a, size)
    b = batching.bdim_at_front(b, bd_b, size)
    return eigh_gen_p.bind(a, b,
                           lower=lower,
                           itype=itype,
                           deg_thresh=deg_thresh), (0, 0)

def _eigh_gen_lowering(*args, **kwargs):
    raise NotImplementedError("Generalized eigh is only implemented for CPU.")

eigh_gen_p = Primitive('eigh_gen')
eigh_gen_p.multiple_results = True
eigh_gen_p.def_impl(_eigh_gen_impl)
eigh_gen_p.def_abstract_eval(_eigh_gen_abstract_eval)
ad.primitive_jvps[eigh_gen_p] = _eigh_gen_jvp_rule
batching.primitive_batchers[eigh_gen_p] = _eigh_gen_batching_rule
mlir.register_lowering(eigh_gen_p, _eigh_gen_lowering)
mlir.register_lowering(eigh_gen_p, _eigh_gen_cpu_lowering, platform='cpu')

