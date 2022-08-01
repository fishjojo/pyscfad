import warnings
from jax import tree_util
from pyscf import __config__

PYSCFAD = getattr(__config__, "pyscfad", False)

def pytree_node(leaf_names, num_args=0):
    '''
    Class decorator that register the underlying class as a pytree node.
    See `jax document<https://jax.readthedocs.io/en/latest/pytrees.html>`_
    for the definition of pytrees.

    Args:
        leaf_names : list or tuple
            Attributes of the class that are traced as pytree leaves.

    Kwargs:
        num_args: int
            Number of positional arguments in ``leaf_names``.
            This is useful when the ``__init__`` method of the class
            has positional arguments that are named differently than
            the actual attribute names. Default value is 0.

    Note:
        The ``__init__`` method of the class can't have positional arguments
        that are not included in ``leaf_names``. If ``num_args`` is greater
        than 0, the sequence of positional arguments in ``leaf_names`` must
        follow that in the ``__init__`` method.
    '''
    def class_orig(cls):
        return cls

    def class_as_pytree_node(cls):
        def tree_flatten(obj):
            keys = obj.__dict__.keys()
            for leaf_name in leaf_names:
                if leaf_name not in keys:
                    raise KeyError(f"Pytree leaf {leaf_name} is not defined in class {cls}.")
            children =  tuple(getattr(obj, leaf_name, None) for leaf_name in leaf_names)
            if len(children) <= 0:
                #raise KeyError("Empty pytree node is not supported.")
                warnings.warn(f"Not taking derivatives wrt the leaves in "
                              f"the node {obj.__class__} as none of those was specified.")

            aux_keys = set(keys) - set(leaf_names)
            aux_data = tuple(getattr(obj, key, None) for key in aux_keys)
            metadata = (num_args,) + tuple(zip(aux_keys, aux_data))
            return children, metadata

        def tree_unflatten(metadata, children):
            num_args = metadata[0]
            metadata = metadata[1:]
            leaves_args = children[:num_args]
            leaves_kwargs = tuple(zip(leaf_names[num_args:], children[num_args:]))
            kwargs = dict(leaves_kwargs + metadata)
            obj = cls(*leaves_args, **kwargs)
            return obj

        tree_util.register_pytree_node(cls, tree_flatten, tree_unflatten)
        return cls

    if PYSCFAD:
        return class_as_pytree_node
    else:
        return class_orig
