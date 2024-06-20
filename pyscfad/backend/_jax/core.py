import warnings
import jax
from jax import numpy as jnp
from jax import tree_util

def is_array(x):
    return isinstance(x, jax.Array)

def to_numpy(x):
    x = jax.lax.stop_gradient(x)
    return x.__array__()

def convert_to_tensor(x, dtype=None, **kwargs):
    return jnp.asarray(x, dtype=dtype, **kwargs)

def vmap(fun, in_axes=0, out_axes=0, chunk_size=None, signature=None):
    return jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)

# TODO deprecate these
def index_update(x, idx, y):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return x.at[idx].set(y)

def index_add(x, idx, y):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return x.at[idx].add(y)

def index_mul(x, idx, y):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return x.at[idx].multiply(y)


def _dict_hash(this):
    from pyscf.lib.misc import finger
    fg = []
    leaves, tree = tree_util.tree_flatten(this)
    fg.append(hash(tree))
    for v in leaves:
        if hasattr(v, 'size'): # arrays
            fg.append(finger(v))
        elif isinstance(v, set):
            fg.append(_dict_hash(tuple(v)))
        else:
            try:
                fg.append(hash(v))
            except TypeError as e:
                raise e
    return hash(tuple(fg))


def _dict_equality(d1, d2):
    leaves1, tree1 = tree_util.tree_flatten(d1)
    leaves2, tree2 = tree_util.tree_flatten(d2)
    if tree1 != tree2:
        return False

    for v1, v2 in zip(leaves1, leaves2):
        if v1 is v2:
            neq = False
        else:
            if hasattr(v1, 'size') and hasattr(v2, 'size'): # arrays
                if v1.size != v2.size:
                    neq = True
                elif v1.size == 0 and v2.size == 0:
                    neq = False
                else:
                    try:
                        neq = not v1 == v2
                    except ValueError as e:
                        try:
                            neq = not (v1 == v2).all()
                        except Exception:
                            raise e from None
            else:
                try:
                    neq = not v1 == v2
                except ValueError as e:
                    raise e
        if neq:
            return False
    return True


class _AuxData:
    def __init__(self, **kwargs):
        self.data = {**kwargs}

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, _AuxData):
            return False
        return _dict_equality(self.data, other.data)

    def __hash__(self):
        return _dict_hash(self.data)


def class_as_pytree_node(cls, leaf_names, num_args=0):
    def tree_flatten(obj):
        keys = obj.__dict__.keys()
        for leaf_name in leaf_names:
            if leaf_name not in keys:
                raise KeyError(f'Pytree leaf {leaf_name} is not defined in class {cls}.')
        children =  tuple(getattr(obj, leaf_name, None) for leaf_name in leaf_names)
        if len(children) <= 0:
            #raise KeyError("Empty pytree node is not supported.")
            warnings.warn(f'Not taking derivatives w.r.t. the leaves in '
                          f'the node {obj.__class__} as none of those was specified.')

        aux_keys = list(set(keys) - set(leaf_names))
        aux_data = list(getattr(obj, key, None) for key in aux_keys)
        metadata = (num_args,) + (_AuxData(**dict(zip(aux_keys, aux_data))),)
        return children, metadata

    def tree_unflatten(metadata, children):
        num_args = metadata[0]
        auxdata = metadata[1]
        leaves_args = children[:num_args]
        leaves_kwargs = dict(zip(leaf_names[num_args:], children[num_args:]))
        kwargs = {**leaves_kwargs, **(auxdata.data)}
        obj = cls(*leaves_args, **kwargs)
        return obj

    tree_util.register_pytree_node(cls, tree_flatten, tree_unflatten)
    return cls
