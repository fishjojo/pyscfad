from functools import partial
from pyscfad import ops

def pytree_node(leaf_names, num_args=0, exclude_aux_name=()):
    """Class decorator that registers the underlying class as a pytree.

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

def to_pyscf(obj, nocopy_names=(), out=None):
    """Convert the pyscfad object to its pyscf counterpart.

    The conversion effectively removes the tracing of the object
    and its members.

    Parameters
    ----------
    obj : object
        The pyscfad object to be converted.
    nocopy_names : tuple, default=()
        Names of attributes that are not copied to the pyscf object.
    out : object, optional
        The target pyscf object.

    Notes
    -----
    Member arrays will be converted (whether a copy is made depends on
    the implementation of ``__array__`` function) to numpy arrays.
    """
    if obj.__module__.startswith("pyscf."):
        return obj

    if out is None:
        from importlib import import_module
        from pyscf.lib.misc import omniobj
        mod = import_module(obj.__module__.replace("pyscfad", "pyscf"))
        cls = getattr(mod, obj.__class__.__name__)
        out = cls(omniobj)

    cls_keys = [getattr(cls, "_keys", ()) for cls in out.__class__.__mro__[:-1]]
    out_keys = set(out.__dict__).union(*cls_keys)
    # Only overwrite the attributes of the same name.
    keys = set(obj.__dict__).intersection(out_keys)
    keys = keys - set(nocopy_names)

    for key in keys:
        val = getattr(obj, key)
        if ops.is_array(val):
            val = ops.to_numpy(val)
        elif hasattr(val, "to_pyscf"):
            val = val.to_pyscf()
        setattr(out, key, val)
    return out

def is_tracer(a):
    """Test if the object is an tracer.

    Parameters
    ----------
    a : object
        The object to be tested.

    Notes
    -----
    Only works for the jax backend.
    """
    return any(cls.__name__.endswith("Tracer") for cls in a.__class__.__mro__)
