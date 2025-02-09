from .config import get_backend

__all__ = [
    'PytreeNode',
    'class_as_pytree_node',
]

def __getattr__(name):
    return getattr(get_backend(), name)

def class_as_pytree_node(cls, leaf_names, num_args=0, exclude_aux_name=()):
    return get_backend().class_as_pytree_node(cls, leaf_names, num_args=num_args,
                                              exclude_aux_name=exclude_aux_name)
