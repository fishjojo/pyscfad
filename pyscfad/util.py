from functools import partial
from pyscfad import ops

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
    return partial(ops.class_as_pytree_node, leaf_names=leaf_names, num_args=num_args)
