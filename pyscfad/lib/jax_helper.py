"""
Extensions to dataclasses
"""

import dataclasses
from jax import tree_util

def dataclass(cls):
    data_cls = dataclasses.dataclass()(cls)
    data_fields = []
    meta_fields = []
    for field, field_info in data_cls.__dataclass_fields__.items():
        is_pytree_node = field_info.metadata.get('pytree_node', False)
        if is_pytree_node:
            data_fields.append(field)
        else:
            meta_fields.append(field)

    def tree_flatten(obj):
        data =  tuple(getattr(obj, key, "None") for key in data_fields)
        meta =  tuple(getattr(obj, key, "None") for key in meta_fields)
        return data, meta

    def tree_unflatten(meta, data):
        data_args = tuple(zip(data_fields, data))
        meta_args = tuple(zip(meta_fields, meta))
        kwargs = dict(data_args + meta_args)
        obj = data_cls(**kwargs)
        return obj

    tree_util.register_pytree_node(data_cls,
                                   tree_flatten,
                                   tree_unflatten)
    return data_cls

def field(pytree_node=False, **kwargs):
    return dataclasses.field(metadata={'pytree_node': pytree_node}, **kwargs)
