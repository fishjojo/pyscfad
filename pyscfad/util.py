from functools import partial
from pyscfad import ops

def pytree_node(leaf_names, num_args=0, exclude_aux_name=()):
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
    exclude_aux_name : tuple, default=()
        A set of static attribute names that are not used for comparing
        the pytrees. Note that ``jax.jit`` recompiles the function for input
        pytrees with differen static attribute values.

    Notes
    -----
    The ``__init__`` method of the class can't have positional arguments
    that are not included in ``leaf_names``. If ``num_args`` is greater
    than 0, the sequence of positional arguments in ``leaf_names`` must
    follow that in the ``__init__`` method.
    """
    return partial(ops.class_as_pytree_node,
                   leaf_names=leaf_names,
                   num_args=num_args,
                   exclude_aux_name=exclude_aux_name)
