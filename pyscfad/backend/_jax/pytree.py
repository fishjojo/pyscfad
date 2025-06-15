from functools import partial
import warnings
from jax import tree_util

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
    def __init__(self, data, exclude_name=()):
        self.data = data
        self.exclude_name = exclude_name

    @property
    def data_for_hash(self):
        if self.exclude_name:
            return {k : v for k, v in self.data.items() if k not in self.exclude_name}
        else:
            return self.data

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, _AuxData):
            return False
        return _dict_equality(self.data_for_hash, other.data_for_hash)

    def __hash__(self):
        return _dict_hash(self.data_for_hash)


def class_as_pytree_node(cls, leaf_names, num_args=0, exclude_aux_name=()):
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
        aux_data = {k : getattr(obj, k, None) for k in aux_keys}
        metadata = (num_args, _AuxData(aux_data, exclude_name=exclude_aux_name))
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


class PytreeNodeMeta(type):
    def __new__(mcls, name, bases, dct, **kwargs):
        cls = super().__new__(mcls, name, bases, dct, **kwargs)

        # preserve the order of the attributes
        _dynamic_attr = []
        for base in reversed(cls.__mro__):
            if hasattr(base, '_dynamic_attr'):
                _dynamic_attr.extend(base._dynamic_attr)
        _dynamic_attr = tuple(dict.fromkeys(_dynamic_attr))

        def _flatten(obj, keys=(), with_keys=False):
            if keys:
                if with_keys:
                    children = [(tree_util.GetAttrKey(key), getattr(obj, key, None)) for key in keys]
                else:
                    children = [getattr(obj, key, None) for key in keys]
            else:
                children = []

            aux_keys = set(obj.__dict__.keys()) - set(keys)
            aux_data = {key: getattr(obj, key) for key in aux_keys}
            return children, _AuxData(aux_data)


        def _unflatten(cls, keys, aux_data, children):
            obj = object.__new__(cls)
            for key, value in zip(keys, children):
                object.__setattr__(obj, key, value)
            for key, value in aux_data.data.items():
                object.__setattr__(obj, key, value)
            return obj

        flatten_with_keys = partial(_flatten, keys=_dynamic_attr, with_keys=True)
        flatten_func = partial(_flatten, keys=_dynamic_attr, with_keys=False)
        unflatten_func = partial(_unflatten, cls, _dynamic_attr)

        tree_util.register_pytree_with_keys(
            cls,
            flatten_with_keys,
            unflatten_func,
            flatten_func=flatten_func
        )
        return cls


class PytreeNode(metaclass=PytreeNodeMeta):
    pass

