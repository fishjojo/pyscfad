"""
Helper functions for jax
"""

import dataclasses
import numpy
import jax
from jax import tree_util
from pyscf import __config__

PYSCFAD = getattr(__config__, "pyscfad", False)

def stop_grad(x):
    if PYSCFAD:
        return jax.lax.stop_gradient(x)
    else:
        return x

def jit(fun, **kwargs):
    if PYSCFAD:
        return jax.jit(fun, **kwargs)
    else:
        return fun

def vmap_numpy(fun, in_axes=0, out_axes=0, axis_name=None, axis_size=None, signature=None):
    if axis_name is not None:
        raise NotImplementedError
    if axis_size is not None:
        raise NotImplementedError
    if not isinstance(out_axes, int):
        raise NotImplementedError

    def vmap_f(*args):
        if isinstance(in_axes, int):
            in_axes_loc = (in_axes,) * len(args)
        else:
            in_axes_loc = in_axes

        if isinstance(in_axes_loc, (list, tuple)):
            excluded = []
            vmap_args = []
            assert len(in_axes_loc) == len(args)
            for i, axis in enumerate(in_axes_loc):
                if axis is None:
                    excluded.append(i)
                    vmap_args.append(args[i])
                elif isinstance(axis, int):
                    vmap_args.append(numpy.moveaxis(args[i], axis, 0))
                else:
                    raise KeyError
            if len(excluded) > 0:
                excluded = set(excluded)
            else:
                excluded = None

            vfun = numpy.vectorize(fun, excluded=excluded, signature=signature)
            out = vfun(*vmap_args)
        else:
            raise KeyError

        if out_axes != 0:
            out = numpy.moveaxis(out, 0, out_axes)
        return out

    return vmap_f

if PYSCFAD:
    def vmap(fun, in_axes=0, out_axes=0, axis_name=None, axis_size=None, signature=None):
        f_vmap = jax.vmap(fun, in_axes=in_axes, out_axes=out_axes,
                          axis_name=axis_name, axis_size=axis_size)
        return f_vmap
else:
    vmap = vmap_numpy

if PYSCFAD:
    custom_jvp = jax.custom_jvp
else:
    class custom_jvp():
        '''
        A fake custom_jvp that does nothing
        '''
        def __init__(self, fun, *args, **kwargs):
            self.fun = fun
            self.jvp = None

        def defjvp(self, jvp):
            self.jvp = jvp
            return jvp

        def __call__(self, *args, **kwargs):
            return self.fun(*args, **kwargs)


def dataclass(cls):
    data_cls = dataclasses.dataclass()(cls)
    data_fields = []
    meta_fields = []
    for field_name, field_info in data_cls.__dataclass_fields__.items():
        is_pytree_node = field_info.metadata.get('pytree_node', False)
        if is_pytree_node:
            data_fields.append(field_name)
        else:
            meta_fields.append(field_name)

    def tree_flatten(obj):
        children =  tuple(getattr(obj, key, None) for key in data_fields)
        metadata =  tuple(getattr(obj, key, None) for key in meta_fields)
        return children, metadata

    def tree_unflatten(metadata, children):
        data_args = tuple(zip(data_fields, children))
        meta_args = tuple(zip(meta_fields, metadata))
        kwargs = dict(data_args + meta_args)
        obj = data_cls(**kwargs)
        return obj

    tree_util.register_pytree_node(data_cls,
                                   tree_flatten,
                                   tree_unflatten)
    return data_cls

def field(pytree_node=False, **kwargs):
    return dataclasses.field(metadata={'pytree_node': pytree_node}, **kwargs)
