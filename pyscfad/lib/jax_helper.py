"""
Helper functions for jax
"""

import dataclasses
import jax
from jax import tree_util

stop_grad = jax.lax.stop_gradient

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
