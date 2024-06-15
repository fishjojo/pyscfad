import warnings
from jax import tree_util
from pyscfad import backend

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


def pytree_node(leaf_names, num_args=0):
    """Class decorator that registers the underlying class as a pytree node.

    See `jax document <https://jax.readthedocs.io/en/latest/pytrees.html>`_
    for the definition of pytrees.

    Parameters
    ----------
    leaf_names : list or tuple
        Attributes of the class that are traced as pytree leaves.
    num_args : int, optional
        Number of positional arguments in ``leaf_names``.
        This is useful when the ``__init__`` method of the class
        has positional arguments that are named differently than
        the actual attribute names. Default value is 0.

    Notes
    -----
    The ``__init__`` method of the class can't have positional arguments
    that are not included in ``leaf_names``. If ``num_args`` is greater
    than 0, the sequence of positional arguments in ``leaf_names`` must
    follow that in the ``__init__`` method.
    """
    def class_orig(cls):
        return cls

    def class_as_pytree_node(cls):
        def tree_flatten(obj):
            keys = obj.__dict__.keys()
            for leaf_name in leaf_names:
                if leaf_name not in keys:
                    raise KeyError(f'Pytree leaf {leaf_name} is not defined in class {cls}.')
            children =  tuple(getattr(obj, leaf_name, None) for leaf_name in leaf_names)
            if len(children) <= 0:
                #raise KeyError("Empty pytree node is not supported.")
                warnings.warn(f'Not taking derivatives wrt the leaves in '
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

    if backend.backend() == 'jax':
        return class_as_pytree_node
    else:
        return class_orig
